package collector

import (
	"context"
	"math/rand/v2"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

type FakeCollector struct {
	nodeID string
}

func NewFakeCollector(nodeID string) *FakeCollector {
	return &FakeCollector{nodeID: nodeID}
}

func (f *FakeCollector) Name() string       { return "fake" }
func (f *FakeCollector) SourceType() string { return "fake" }

func (f *FakeCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	now := time.Now()
	return []model.RawSample{
		{
			Timestamp:  now,
			SourceType: "fake",
			EntityType: "cpu",
			EntityID:   f.nodeID + ":cpu0",
			MetricName: "cpu_usage_total_percent",
			Value:      10 + rand.Float64()*80,
			Unit:       "percent",
		},
		{
			Timestamp:  now,
			SourceType: "fake",
			EntityType: "ram",
			EntityID:   f.nodeID + ":ram0",
			MetricName: "ram_used_percent",
			Value:      20 + rand.Float64()*60,
			Unit:       "percent",
		},
		{
			Timestamp:  now,
			SourceType: "fake",
			EntityType: "cpu",
			EntityID:   f.nodeID + ":cpu0",
			MetricName: "cpu_temperature_celsius",
			Value:      40 + rand.Float64()*30,
			Unit:       "celsius",
		},
	}, nil
}
