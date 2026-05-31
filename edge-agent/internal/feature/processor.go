package feature

import (
	"strings"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

// categoricalSuffixes — blocklist per §1.4. Any metric name ending with one
// of these is categorical; everything else gets 4-channel temporal processing.
var categoricalSuffixes = []string{"_encoded", "_flag", "_status"}

func isCategorical(metricName string) bool {
	for _, suf := range categoricalSuffixes {
		if strings.HasSuffix(metricName, suf) {
			return true
		}
	}
	return false
}

// Processor computes temporal features (delta_short, delta_long, rolling_var)
// for each continuous metric and passes categorical metrics through unchanged.
type Processor struct {
	store *Store
}

func NewProcessor(longWindow int, emaAlpha float64) *Processor {
	return &Processor{store: NewStore(longWindow, emaAlpha)}
}

// Process converts a slice of RawSamples into ProcessedSamples.
func (p *Processor) Process(samples []model.RawSample) []model.ProcessedSample {
	out := make([]model.ProcessedSample, len(samples))
	for i, s := range samples {
		out[i] = p.processSample(s)
	}
	return out
}

func (p *Processor) processSample(s model.RawSample) model.ProcessedSample {
	ps := model.ProcessedSample{RawSample: s}

	if isCategorical(s.MetricName) {
		ps.IsCategorical = true
		// DeltaShort, DeltaLong, RollingVar stay 0.
		return ps
	}

	key := s.EntityID + "|" + s.MetricName
	ps.DeltaShort, ps.DeltaLong, ps.RollingVar = p.store.Update(key, s.Value)
	return ps
}
