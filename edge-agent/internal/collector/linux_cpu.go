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

type LinuxCPUCollector struct {
	nodeID string
	mu     sync.Mutex
	prev   *cpuStat
	prevTs time.Time
}

func NewLinuxCPUCollector(nodeID string) *LinuxCPUCollector {
	return &LinuxCPUCollector{nodeID: nodeID}
}

func (c *LinuxCPUCollector) Name() string       { return "linux_cpu" }
func (c *LinuxCPUCollector) SourceType() string { return "linux" }

type cpuStat struct {
	user, nice, system, idle, iowait, irq, softirq, steal uint64
	ctxt                                                  uint64
}

func (s *cpuStat) total() uint64 {
	return s.user + s.nice + s.system + s.idle + s.iowait + s.irq + s.softirq + s.steal
}

func readCPUStat() (*cpuStat, error) {
	f, err := os.Open("/proc/stat")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st := &cpuStat{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "cpu ") {
			fields := strings.Fields(line)
			if len(fields) < 8 {
				continue
			}
			nums := make([]uint64, 0, 8)
			for _, s := range fields[1:9] {
				n, _ := strconv.ParseUint(s, 10, 64)
				nums = append(nums, n)
			}
			st.user, st.nice, st.system, st.idle, st.iowait, st.irq, st.softirq, st.steal =
				nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7]
		}
		if strings.HasPrefix(line, "ctxt ") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				st.ctxt, _ = strconv.ParseUint(fields[1], 10, 64)
			}
		}
	}
	return st, scanner.Err()
}

func readCPUFreqMHz() float64 {
	// Try cpufreq scaling_cur_freq (kHz → MHz).
	p := "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
	if data, err := os.ReadFile(p); err == nil {
		if khz, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64); err == nil {
			return khz / 1000
		}
	}
	// Fallback: cpuinfo_max_freq.
	p = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
	if data, err := os.ReadFile(p); err == nil {
		if khz, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64); err == nil {
			return khz / 1000
		}
	}
	return 0
}

func readCPUTempCelsius() float64 {
	// Walk hwmon devices looking for a CPU temperature sensor.
	dirs, _ := filepath.Glob("/sys/class/hwmon/hwmon*/")
	for _, d := range dirs {
		nameFile := filepath.Join(d, "name")
		if nameBytes, err := os.ReadFile(nameFile); err == nil {
			name := strings.TrimSpace(string(nameBytes))
			if strings.Contains(name, "coretemp") || strings.Contains(name, "k10temp") || strings.Contains(name, "cpu") {
				if temps, _ := filepath.Glob(filepath.Join(d, "temp*_input")); len(temps) > 0 {
					if data, err := os.ReadFile(temps[0]); err == nil {
						if milli, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64); err == nil {
							return milli / 1000
						}
					}
				}
			}
		}
	}
	// Fallback: thermal_zone0.
	if data, err := os.ReadFile("/sys/class/thermal/thermal_zone0/temp"); err == nil {
		if milli, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64); err == nil {
			return milli / 1000
		}
	}
	return 0
}

// readCPUPowerWatts reads instantaneous CPU package power via Intel RAPL.
// Returns 0 if unavailable.
func readCPUPowerWatts() float64 {
	p := "/sys/class/powercap/intel-rapl/intel-rapl:0/power_uw"
	if data, err := os.ReadFile(p); err == nil {
		if uw, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64); err == nil {
			return uw / 1e6
		}
	}
	return 0
}

func (c *LinuxCPUCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	cur, err := readCPUStat()
	if err != nil {
		return nil, fmt.Errorf("read /proc/stat: %w", err)
	}
	now := time.Now()

	c.mu.Lock()
	prev := c.prev
	prevTs := c.prevTs
	c.prev = cur
	c.prevTs = now
	c.mu.Unlock()

	entityID := c.nodeID + ":cpu0"

	// First call — no deltas yet.
	if prev == nil {
		return []model.RawSample{
			metric(now, "linux", "cpu", entityID, "cpu_frequency_mhz", readCPUFreqMHz(), "mhz"),
			metric(now, "linux", "cpu", entityID, "cpu_temperature_celsius", readCPUTempCelsius(), "celsius"),
			metric(now, "linux", "cpu", entityID, "cpu_power_watts", readCPUPowerWatts(), "watts"),
		}, nil
	}

	dt := now.Sub(prevTs).Seconds()
	if dt <= 0 {
		dt = 1
	}

	dtotal := cur.total() - prev.total()
	if dtotal == 0 {
		dtotal = 1
	}

	pct := func(d uint64) float64 { return float64(d) / float64(dtotal) * 100 }
	duser := cur.user - prev.user
	dsystem := cur.system - prev.system
	didle := cur.idle - prev.idle
	diowait := cur.iowait - prev.iowait

	ctxPs := float64(cur.ctxt-prev.ctxt) / dt

	samples := []model.RawSample{
		metric(now, "linux", "cpu", entityID, "cpu_usage_total_percent", pct(dtotal-didle-diowait), "percent"),
		metric(now, "linux", "cpu", entityID, "cpu_user_percent", pct(duser), "percent"),
		metric(now, "linux", "cpu", entityID, "cpu_system_percent", pct(dsystem), "percent"),
		metric(now, "linux", "cpu", entityID, "cpu_iowait_percent", pct(diowait), "percent"),
		metric(now, "linux", "cpu", entityID, "cpu_frequency_mhz", readCPUFreqMHz(), "mhz"),
		metric(now, "linux", "cpu", entityID, "cpu_temperature_celsius", readCPUTempCelsius(), "celsius"),
		metric(now, "linux", "cpu", entityID, "cpu_power_watts", readCPUPowerWatts(), "watts"),
		metric(now, "linux", "cpu", entityID, "cpu_context_switches_ps", ctxPs, "count"),
	}
	return samples, nil
}

// metric is a small helper to build a RawSample without Labels.
func metric(ts time.Time, src, etype, eid, name string, val float64, unit string) model.RawSample {
	return model.RawSample{
		Timestamp:  ts,
		SourceType: src,
		EntityType: etype,
		EntityID:   eid,
		MetricName: name,
		Value:      val,
		Unit:       unit,
	}
}
