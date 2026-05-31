//go:build linux

package collector

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

// cgroupJobStat holds per-job counters between ticks for rate computation.
type cgroupJobStat struct {
	cpuUsageNs   uint64
	ioReadBytes  uint64
	ioWriteBytes uint64
	netRxBytes   uint64
	netTxBytes   uint64
	ts           time.Time
}

// CgroupsSLURMCollector reads cgroup v2 accounting for SLURM jobs.
// It maps cgroup slices to job_id via /proc/<pid>/cgroup + SLURM env vars.
type CgroupsSLURMCollector struct {
	nodeID string
	mu     sync.Mutex
	prev   map[string]cgroupJobStat // key: job_id
}

func NewCgroupsSLURMCollector(nodeID string) *CgroupsSLURMCollector {
	return &CgroupsSLURMCollector{
		nodeID: nodeID,
		prev:   make(map[string]cgroupJobStat),
	}
}

func (c *CgroupsSLURMCollector) Name() string       { return "cgroups_slurm" }
func (c *CgroupsSLURMCollector) SourceType() string { return "cgroups" }

// slurmJobRoot tries common cgroup v2 paths for SLURM.
func slurmJobRoot() string {
	for _, p := range []string{
		"/sys/fs/cgroup/system.slice/slurmstepd.scope",
		"/sys/fs/cgroup/slurm",
		"/sys/fs/cgroup/user.slice",
	} {
		if fi, err := os.Stat(p); err == nil && fi.IsDir() {
			return p
		}
	}
	return ""
}

// findSlurmJobDirs walks under root looking for directories whose name starts
// with "job_" (SLURM v22+ cgroup v2 naming) and returns jobID → cgroupPath.
func findSlurmJobDirs(root string) map[string]string {
	jobs := make(map[string]string)
	if root == "" {
		return jobs
	}
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, "job_") {
			jobID := strings.TrimPrefix(name, "job_")
			jobs[jobID] = path
		}
		return nil
	})
	return jobs
}

func readCgroupUint64(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

// readCgroupCPUUsageNs reads cpu.stat's usage_usec (µs → ns).
func readCgroupCPUUsageNs(cgPath string) uint64 {
	f, err := os.Open(filepath.Join(cgPath, "cpu.stat"))
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) == 2 && parts[0] == "usage_usec" {
			v, _ := strconv.ParseUint(parts[1], 10, 64)
			return v * 1000 // µs → ns
		}
	}
	return 0
}

// readCgroupIO reads io.stat for rbytes and wbytes (aggregated across all devices).
func readCgroupIO(cgPath string) (rbytes, wbytes uint64) {
	f, err := os.Open(filepath.Join(cgPath, "io.stat"))
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		for _, field := range strings.Fields(scanner.Text()) {
			kv := strings.SplitN(field, "=", 2)
			if len(kv) != 2 {
				continue
			}
			v, _ := strconv.ParseUint(kv[1], 10, 64)
			switch kv[0] {
			case "rbytes":
				rbytes += v
			case "wbytes":
				wbytes += v
			}
		}
	}
	return
}

// readCgroupMemUsedBytes reads memory.current (current usage in bytes).
func readCgroupMemUsedBytes(cgPath string) uint64 {
	v, _ := readCgroupUint64(filepath.Join(cgPath, "memory.current"))
	return v
}

// readCgroupMemLimitBytes reads memory.max ("max" means unlimited → 0).
func readCgroupMemLimitBytes(cgPath string) uint64 {
	data, err := os.ReadFile(filepath.Join(cgPath, "memory.max"))
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(data))
	if s == "max" {
		return 0
	}
	v, _ := strconv.ParseUint(s, 10, 64)
	return v
}

// slurmJobMeta reads job metadata from the SLURM environment of any process
// inside the cgroup. Returns jobID, priority, waitSecs (best-effort; 0 on failure).
func slurmJobMeta(cgPath string) (priority, waitSecs float64) {
	// Read pids from cgroup.procs and inspect /proc/<pid>/environ for SLURM vars.
	procsPath := filepath.Join(cgPath, "cgroup.procs")
	data, err := os.ReadFile(procsPath)
	if err != nil {
		return
	}

	for _, pidStr := range strings.Fields(string(data)) {
		envPath := fmt.Sprintf("/proc/%s/environ", pidStr)
		envData, err := os.ReadFile(envPath)
		if err != nil {
			continue
		}
		env := make(map[string]string)
		for _, entry := range strings.Split(string(envData), "\x00") {
			kv := strings.SplitN(entry, "=", 2)
			if len(kv) == 2 {
				env[kv[0]] = kv[1]
			}
		}
		if p, ok := env["SLURM_JOB_NICE"]; ok {
			if v, err := strconv.ParseFloat(p, 64); err == nil {
				priority = v
			}
		}
		// SLURM_JOB_START_TIME is epoch seconds; wait = start - submit.
		if sub, ok1 := env["SLURM_JOB_SUBMIT_TIME"]; ok1 {
			if start, ok2 := env["SLURM_JOB_START_TIME"]; ok2 {
				subV, _ := strconv.ParseFloat(sub, 64)
				startV, _ := strconv.ParseFloat(start, 64)
				if startV > subV {
					waitSecs = startV - subV
				}
			}
		}
		return
	}
	return
}

// cpuIDsForJob returns a comma-separated list of CPU IDs from cpuset.cpus.
func cpuIDsForJob(cgPath string) string {
	data, err := os.ReadFile(filepath.Join(cgPath, "cpuset.cpus"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func (c *CgroupsSLURMCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	root := slurmJobRoot()
	jobs := findSlurmJobDirs(root)

	now := time.Now()

	c.mu.Lock()
	prev := c.prev
	newPrev := make(map[string]cgroupJobStat, len(jobs))
	c.mu.Unlock()

	var samples []model.RawSample

	for jobID, cgPath := range jobs {
		cpuNs := readCgroupCPUUsageNs(cgPath)
		ioR, ioW := readCgroupIO(cgPath)
		memUsed := readCgroupMemUsedBytes(cgPath)
		memLimit := readCgroupMemLimitBytes(cgPath)
		priority, waitSecs := slurmJobMeta(cgPath)
		cpuIDs := cpuIDsForJob(cgPath)

		var cpuPct, ioReadMBs, ioWriteMBs float64
		if ps, ok := prev[jobID]; ok {
			dt := now.Sub(ps.ts).Seconds()
			if dt > 0 {
				cpuPct = float64(cpuNs-ps.cpuUsageNs) / (dt * 1e9) * 100
				ioReadMBs = float64(ioR-ps.ioReadBytes) / 1024 / 1024 / dt
				ioWriteMBs = float64(ioW-ps.ioWriteBytes) / 1024 / 1024 / dt
			}
		}

		newPrev[jobID] = cgroupJobStat{
			cpuUsageNs:   cpuNs,
			ioReadBytes:  ioR,
			ioWriteBytes: ioW,
			ts:           now,
		}

		var ramPct float64
		if memLimit > 0 {
			ramPct = float64(memUsed) / float64(memLimit) * 100
		}

		eid := c.nodeID + ":job" + jobID
		labels := map[string]string{
			"job_id":  jobID,
			"cpu_ids": cpuIDs,
		}

		add := func(name string, val float64, unit string) {
			s := metric(now, "cgroups", "job", eid, name, val, unit)
			s.Labels = labels
			samples = append(samples, s)
		}

		// runtime = wall time since cgroup creation (approximate via mtime of cgroup dir)
		var runtimeSecs float64
		if fi, err := os.Stat(cgPath); err == nil {
			runtimeSecs = now.Sub(fi.ModTime()).Seconds()
		}

		add("job_cpu_usage_percent", cpuPct, "percent")
		add("job_gpu_usage_percent", 0, "percent") // GPU: reported separately by nvml
		add("job_ram_usage_percent", ramPct, "percent")
		add("job_io_read_mb", ioReadMBs, "mb")
		add("job_io_write_mb", ioWriteMBs, "mb")
		add("job_network_rx_mb", 0, "mb") // network: reported by fabric collector
		add("job_network_tx_mb", 0, "mb")
		add("job_runtime_seconds", runtimeSecs, "seconds")
		add("job_wait_time_seconds", waitSecs, "seconds")
		add("job_priority_encoded", priority, "encoded")
		add("job_status_encoded", 1, "encoded") // 1 = running
	}

	c.mu.Lock()
	c.prev = newPrev
	c.mu.Unlock()

	return samples, nil
}
