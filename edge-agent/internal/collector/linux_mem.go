//go:build linux

package collector

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

type LinuxMemCollector struct {
	nodeID string
	mu     sync.Mutex
	prevPF uint64
	prevTs time.Time
}

func NewLinuxMemCollector(nodeID string) *LinuxMemCollector {
	return &LinuxMemCollector{nodeID: nodeID}
}

func (c *LinuxMemCollector) Name() string       { return "linux_mem" }
func (c *LinuxMemCollector) SourceType() string { return "linux" }

func parseMeminfo() (map[string]uint64, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := make(map[string]uint64, 32)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		key := strings.TrimSuffix(parts[0], ":")
		val, _ := strconv.ParseUint(parts[1], 10, 64) // value in kB
		m[key] = val
	}
	return m, scanner.Err()
}

func readVMStatField(field string) (uint64, error) {
	f, err := os.Open("/proc/vmstat")
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) == 2 && parts[0] == field {
			v, _ := strconv.ParseUint(parts[1], 10, 64)
			return v, nil
		}
	}
	return 0, fmt.Errorf("field %q not found in /proc/vmstat", field)
}

// buddyFragScore computes a simple fragmentation score from /proc/buddyinfo:
// ratio of pages in small orders (order 0–3) to total free pages.
func buddyFragScore() float64 {
	f, err := os.Open("/proc/buddyinfo")
	if err != nil {
		return 0
	}
	defer f.Close()

	var small, total float64
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		// Format: Node N, zone ZoneName  n0 n1 n2 ... n10
		if len(fields) < 6 {
			continue
		}
		for i, s := range fields[4:] {
			n, err := strconv.ParseFloat(s, 64)
			if err != nil {
				continue
			}
			pages := n * float64(uint64(1)<<uint(i))
			total += pages
			if i <= 3 {
				small += pages
			}
		}
	}
	if total == 0 {
		return 0
	}
	return small / total
}

func (c *LinuxMemCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	mi, err := parseMeminfo()
	if err != nil {
		return nil, fmt.Errorf("read /proc/meminfo: %w", err)
	}

	curPF, _ := readVMStatField("pgfault")
	now := time.Now()

	c.mu.Lock()
	prevPF := c.prevPF
	prevTs := c.prevTs
	c.prevPF = curPF
	c.prevTs = now
	c.mu.Unlock()

	memTotal := mi["MemTotal"]
	memAvail := mi["MemAvailable"]
	memFree := mi["MemFree"]
	cached := mi["Cached"] + mi["Buffers"]
	swapTotal := mi["SwapTotal"]
	swapFree := mi["SwapFree"]

	var usedPct float64
	if memTotal > 0 {
		usedPct = float64(memTotal-memAvail) / float64(memTotal) * 100
	}
	var swapPct float64
	if swapTotal > 0 {
		swapPct = float64(swapTotal-swapFree) / float64(swapTotal) * 100
	}

	var pfps float64
	dt := now.Sub(prevTs).Seconds()
	if prevTs.IsZero() || dt <= 0 {
		dt = 1
	}
	if !prevTs.IsZero() {
		pfps = float64(curPF-prevPF) / dt
	}

	// ram_bandwidth_mb: rough estimate from pgpgin/pgpgout (pages in/out * 4 KB / 1 MB).
	pgIn, _ := readVMStatField("pgpgin")
	pgOut, _ := readVMStatField("pgpgout")
	_ = pgIn
	_ = pgOut
	// We'd need deltas for bandwidth; leave as 0 on first call.
	bwMB := 0.0

	_ = memFree // available for future use

	eid := c.nodeID + ":ram0"
	samples := []model.RawSample{
		metric(now, "linux", "ram", eid, "ram_used_percent", usedPct, "percent"),
		metric(now, "linux", "ram", eid, "ram_available_mb", float64(memAvail)/1024, "mb"),
		metric(now, "linux", "ram", eid, "ram_cached_mb", float64(cached)/1024, "mb"),
		metric(now, "linux", "ram", eid, "ram_swap_used_percent", swapPct, "percent"),
		metric(now, "linux", "ram", eid, "ram_bandwidth_mb", bwMB, "mb"),
		metric(now, "linux", "ram", eid, "ram_latency_ns", 0, "ns"),
		metric(now, "linux", "ram", eid, "ram_page_faults_ps", pfps, "count"),
		metric(now, "linux", "ram", eid, "ram_fragmentation_score", buddyFragScore(), "ratio"),
	}
	return samples, nil
}
