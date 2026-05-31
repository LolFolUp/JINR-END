//go:build linux

package collector

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

// PMemCollector collects Intel Optane Persistent Memory (PMEM) telemetry for
// H04-class nodes, reading the libnvdimm sysfs tree at /sys/bus/nd/devices/.
//
// This is the "системные интерфейсы" path the diploma (3.1) allows alongside
// ndctl/ipmctl: no fork/exec, just sysfs reads — consistent with the agent's
// zero-overhead philosophy (Table 1.1).
//
// Metrics fold into the unified "ram" node (the diploma's "ram / pmem" node
// type), so they attach to the same entity_id as LinuxMemCollector:
//   pmem_health_status   — 0 healthy, 1 degraded (worst across DIMMs)
//   pmem_mode_flag       — namespace mode: 0 raw, 1 sector, 2 fsdax, 3 devdax
//   pmem_media_read_mb   — cumulative media reads  (0 if unavailable via sysfs)
//   pmem_media_write_mb  — cumulative media writes (0 if unavailable via sysfs)
//
// Media read/write counters live in the DSM SMART payload, reachable only via
// the ndctl ioctl; sysfs does not expose them, so they are reported as 0 with a
// label flag. Health and mode ARE in sysfs and are reported accurately.
//
// On nodes without PMEM (the /sys/bus/nd tree is absent) Collect returns no
// samples and no error — the collector is safe to enable cluster-wide.
type PMemCollector struct {
	nodeID  string
	ndBase  string // overridable for tests; defaults to /sys/bus/nd/devices
}

func NewPMemCollector(nodeID string) *PMemCollector {
	return &PMemCollector{nodeID: nodeID, ndBase: "/sys/bus/nd/devices"}
}

func (c *PMemCollector) Name() string       { return "pmem" }
func (c *PMemCollector) SourceType() string { return "pmem" }

// pmemModeFlag encodes the namespace access mode as an ordinal categorical flag.
func pmemModeFlag(mode string) float64 {
	switch strings.TrimSpace(mode) {
	case "raw":
		return 0
	case "sector", "safe":
		return 1
	case "fsdax", "memory":
		return 2
	case "devdax", "dax":
		return 3
	default:
		return 0
	}
}

// readTrim returns the trimmed contents of a sysfs file, or "" on any error.
func readTrim(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// nmemDegraded reports whether an NVDIMM exposes any active fault flag.
// Intel NFIT platforms expose nmemN/nfit/flags; PAPR (POWER) uses nmemN/papr/flags.
// A non-empty flags string means at least one problem flag is set.
func nmemDegraded(nmemPath string) bool {
	for _, rel := range []string{"nfit/flags", "papr/flags"} {
		if flags := readTrim(filepath.Join(nmemPath, rel)); flags != "" {
			return true
		}
	}
	return false
}

func (c *PMemCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	entries, err := os.ReadDir(c.ndBase)
	if err != nil {
		// No libnvdimm bus on this host → no PMEM. Not an error.
		return nil, nil
	}

	var (
		dimmCount     int
		namespaceN    int
		degraded      bool
		modeFlag      float64
		haveModeFlag  bool
	)

	for _, e := range entries {
		name := e.Name()
		path := filepath.Join(c.ndBase, name)

		switch {
		case strings.HasPrefix(name, "nmem"):
			dimmCount++
			if nmemDegraded(path) {
				degraded = true
			}
		case strings.HasPrefix(name, "namespace"):
			namespaceN++
			// Use the first namespace's mode as the representative node feature.
			if !haveModeFlag {
				if mode := readTrim(filepath.Join(path, "mode")); mode != "" {
					modeFlag = pmemModeFlag(mode)
					haveModeFlag = true
				}
			}
		}
	}

	// No NVDIMMs discovered → nothing to report.
	if dimmCount == 0 && namespaceN == 0 {
		return nil, nil
	}

	now := time.Now()
	eid := c.nodeID + ":ram0" // fold into the unified ram/pmem node
	labels := map[string]string{
		"pmem_dimm_count":      itoa(dimmCount),
		"pmem_namespace_count": itoa(namespaceN),
		"media_counters_src":   "unavailable_sysfs", // need ndctl DSM ioctl
	}

	add := func(name string, val float64, unit string) model.RawSample {
		s := metric(now, "pmem", "ram", eid, name, val, unit)
		s.Labels = labels
		return s
	}

	healthStatus := 0.0
	if degraded {
		healthStatus = 1.0
	}

	return []model.RawSample{
		add("pmem_health_status", healthStatus, "status"),
		add("pmem_mode_flag", modeFlag, "flag"),
		add("pmem_media_read_mb", 0, "mb"),
		add("pmem_media_write_mb", 0, "mb"),
	}, nil
}

// itoa avoids pulling strconv just for label formatting.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
