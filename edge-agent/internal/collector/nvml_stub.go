//go:build !(linux && cgo && nvml)

package collector

import (
	"context"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

// NvmlStubCollector is used when the agent is built without the nvml build tag
// (i.e., no NVIDIA GPU support). It satisfies the Collector interface but
// always returns empty sample slices.
type NvmlStubCollector struct{}

// NewGPUCollector returns the stub when built without -tags nvml.
func NewGPUCollector(_ string) Collector {
	return &NvmlStubCollector{}
}

func (s *NvmlStubCollector) Name() string       { return "nvml_gpu" }
func (s *NvmlStubCollector) SourceType() string { return "nvml" }

func (s *NvmlStubCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	return nil, nil
}
