//go:build linux && cgo && nvml

package collector

import (
	"context"
	"fmt"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/aladed/JINR/edge-agent/internal/model"
)

// NvmlGPUCollector collects GPU telemetry via NVML (requires libnvidia-ml.so).
// Build with: go build -tags nvml ...
type NvmlGPUCollector struct {
	nodeID string
}

// NewGPUCollector returns the real NVML collector when built with -tags nvml.
func NewGPUCollector(nodeID string) Collector {
	return &NvmlGPUCollector{nodeID: nodeID}
}

func (c *NvmlGPUCollector) Name() string       { return "nvml_gpu" }
func (c *NvmlGPUCollector) SourceType() string { return "nvml" }

func (c *NvmlGPUCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		return nil, fmt.Errorf("nvml init: %v", nvml.ErrorString(ret))
	}
	defer nvml.Shutdown()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("nvml device count: %v", nvml.ErrorString(ret))
	}

	now := time.Now()
	var samples []model.RawSample

	for i := range count {
		dev, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			continue
		}

		uuid, _ := nvml.DeviceGetUUID(dev)
		eid := fmt.Sprintf("%s:gpu%d", c.nodeID, i)
		labels := map[string]string{
			"gpu_index": fmt.Sprintf("%d", i),
			"gpu_uuid":  uuid,
		}

		add := func(name string, val float64, unit string) {
			s := metric(now, "nvml", "gpu", eid, name, val, unit)
			s.Labels = labels
			samples = append(samples, s)
		}

		util, ret := nvml.DeviceGetUtilizationRates(dev)
		if ret == nvml.SUCCESS {
			add("gpu_utilization_percent", float64(util.Gpu), "percent")
		}

		memInfo, ret := nvml.DeviceGetMemoryInfo(dev)
		if ret == nvml.SUCCESS {
			add("gpu_memory_used_mb", float64(memInfo.Used)/1024/1024, "mb")
		}

		temp, ret := nvml.DeviceGetTemperature(dev, nvml.TEMPERATURE_GPU)
		if ret == nvml.SUCCESS {
			add("gpu_temperature_celsius", float64(temp), "celsius")
		}

		power, ret := nvml.DeviceGetPowerUsage(dev)
		if ret == nvml.SUCCESS {
			add("gpu_power_watts", float64(power)/1000, "watts")
		}

		clocks, ret := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_SM)
		if ret == nvml.SUCCESS {
			add("gpu_sm_clock_mhz", float64(clocks), "mhz")
		}

		memClock, ret := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_MEM)
		if ret == nvml.SUCCESS {
			add("gpu_memory_clock_mhz", float64(memClock), "mhz")
		}

		// Tensor core utilisation requires DCGM; approximate from SM utilisation.
		if util.Gpu > 0 {
			add("gpu_tensor_utilization_percent", float64(util.Gpu), "percent")
		} else {
			add("gpu_tensor_utilization_percent", 0, "percent")
		}

		pcie, ret := nvml.DeviceGetPcieThroughput(dev, nvml.PCIE_UTIL_TX_BYTES)
		if ret == nvml.SUCCESS {
			add("gpu_pcie_tx_mb", float64(pcie)/1024, "mb")
		}
		pcieRx, ret := nvml.DeviceGetPcieThroughput(dev, nvml.PCIE_UTIL_RX_BYTES)
		if ret == nvml.SUCCESS {
			add("gpu_pcie_rx_mb", float64(pcieRx)/1024, "mb")
		}

		eccCounts, ret := nvml.DeviceGetTotalEccErrors(dev,
			nvml.MEMORY_ERROR_TYPE_UNCORRECTED, nvml.AGGREGATE_ECC)
		if ret == nvml.SUCCESS {
			add("gpu_ecc_errors_count", float64(eccCounts), "count")
		}

		fanSpeed, ret := nvml.DeviceGetFanSpeed(dev)
		if ret == nvml.SUCCESS {
			add("gpu_fan_speed_percent", float64(fanSpeed), "percent")
		}

		// Throttle flag: 1 if any throttle reason active (thermal/power).
		throttleReasons, ret := nvml.DeviceGetCurrentClocksThrottleReasons(dev)
		throttleFlag := 0.0
		if ret == nvml.SUCCESS && throttleReasons != nvml.ClocksThrottleReasonNone {
			throttleFlag = 1
		}
		add("gpu_throttle_flag", throttleFlag, "flag")
	}

	return samples, nil
}
