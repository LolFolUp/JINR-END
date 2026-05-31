//go:build linux

package collector

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

type diskStat struct {
	reads, writes     uint64
	sectorsR, sectorsW uint64
	msReading, msWriting uint64
}

// LinuxDiskCollector reads /proc/diskstats for I/O counters
// and syscall.Statfs for filesystem utilisation.
type LinuxDiskCollector struct {
	nodeID   string
	mu       sync.Mutex
	prev     map[string]diskStat
	prevTs   time.Time
	// device → mount point (built once)
	mounts   map[string]string
}

func NewLinuxDiskCollector(nodeID string) *LinuxDiskCollector {
	return &LinuxDiskCollector{
		nodeID: nodeID,
		prev:   make(map[string]diskStat),
		mounts: buildMounts(),
	}
}

func (c *LinuxDiskCollector) Name() string       { return "linux_disk" }
func (c *LinuxDiskCollector) SourceType() string { return "linux" }

// buildMounts maps device basenames to their mount points via /proc/mounts.
func buildMounts() map[string]string {
	m := make(map[string]string)
	f, err := os.Open("/proc/mounts")
	if err != nil {
		return m
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		dev := fields[0]
		mount := fields[1]
		if strings.HasPrefix(dev, "/dev/") {
			base := dev[len("/dev/"):]
			m[base] = mount
		}
	}
	return m
}

// parseDiskstats reads /proc/diskstats and returns per-device counters.
func parseDiskstats() (map[string]diskStat, error) {
	f, err := os.Open("/proc/diskstats")
	if err != nil {
		return nil, fmt.Errorf("open /proc/diskstats: %w", err)
	}
	defer f.Close()

	result := make(map[string]diskStat)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var (
			major, minor                   int
			name                           string
			reads, readsMerged             uint64
			sectorsR                       uint64
			msR                            uint64
			writes, writesMerged           uint64
			sectorsW                       uint64
			msW                            uint64
			ioInProgress, msIO, msWeighted uint64
		)
		n, err := fmt.Sscanf(scanner.Text(),
			"%d %d %s %d %d %d %d %d %d %d %d %d %d %d",
			&major, &minor, &name,
			&reads, &readsMerged, &sectorsR, &msR,
			&writes, &writesMerged, &sectorsW, &msW,
			&ioInProgress, &msIO, &msWeighted)
		if err != nil || n < 14 {
			continue
		}
		// Skip loop, ram, dm-* if desired — include all physical block devs.
		if strings.HasPrefix(name, "loop") || strings.HasPrefix(name, "ram") {
			continue
		}
		result[name] = diskStat{
			reads:    reads,
			writes:   writes,
			sectorsR: sectorsR,
			sectorsW: sectorsW,
			msReading: msR,
			msWriting: msW,
		}
	}
	return result, scanner.Err()
}

func diskUsagePct(mountPoint string) float64 {
	var st syscall.Statfs_t
	if err := syscall.Statfs(mountPoint, &st); err != nil {
		return 0
	}
	total := st.Blocks * uint64(st.Bsize)
	free := st.Bfree * uint64(st.Bsize)
	if total == 0 {
		return 0
	}
	return float64(total-free) / float64(total) * 100
}

func (c *LinuxDiskCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	cur, err := parseDiskstats()
	if err != nil {
		return nil, err
	}
	now := time.Now()

	c.mu.Lock()
	prev := c.prev
	prevTs := c.prevTs
	c.prev = cur
	c.prevTs = now
	c.mu.Unlock()

	dt := now.Sub(prevTs).Seconds()
	if prevTs.IsZero() || dt <= 0 {
		dt = 1
	}

	var samples []model.RawSample
	for devName, cs := range cur {
		eid := c.nodeID + ":" + devName
		ps, hasPrev := prev[devName]

		mount := c.mounts[devName]
		if mount == "" {
			mount = "/"
		}

		usagePct := diskUsagePct(mount)

		var readMBs, writeMBs, readIops, writeIops, latencyMs float64
		if hasPrev && !prevTs.IsZero() {
			const sectorBytes = 512.0
			readMBs = float64(cs.sectorsR-ps.sectorsR) * sectorBytes / 1024 / 1024 / dt
			writeMBs = float64(cs.sectorsW-ps.sectorsW) * sectorBytes / 1024 / 1024 / dt
			readIops = float64(cs.reads-ps.reads) / dt
			writeIops = float64(cs.writes-ps.writes) / dt

			dReads := cs.reads - ps.reads
			dMs := cs.msReading - ps.msReading
			if dReads > 0 {
				latencyMs = float64(dMs) / float64(dReads)
			}
		}

		samples = append(samples,
			metric(now, "linux", "hdd", eid, "disk_usage_percent", usagePct, "percent"),
			metric(now, "linux", "hdd", eid, "disk_read_mb", readMBs, "mb"),
			metric(now, "linux", "hdd", eid, "disk_write_mb", writeMBs, "mb"),
			metric(now, "linux", "hdd", eid, "disk_read_iops", readIops, "count"),
			metric(now, "linux", "hdd", eid, "disk_write_iops", writeIops, "count"),
			metric(now, "linux", "hdd", eid, "disk_latency_ms", latencyMs, "ms"),
			metric(now, "linux", "hdd", eid, "disk_temperature_celsius", 0, "celsius"),
			metric(now, "linux", "hdd", eid, "disk_reallocated_sectors", 0, "count"),
		)
	}
	return samples, nil
}
