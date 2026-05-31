package feature

import (
	"math"
	"testing"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

const alpha = 0.0645

func TestCategoricalPassThrough(t *testing.T) {
	p := NewProcessor(60, alpha)
	for _, name := range []string{"job_status_encoded", "gpu_throttle_flag", "job_priority_encoded", "link_status_encoded"} {
		samples := []model.RawSample{{EntityID: "e1", MetricName: name, Value: 2}}
		out := p.Process(samples)
		if !out[0].IsCategorical {
			t.Errorf("%s: expected IsCategorical=true", name)
		}
		if out[0].DeltaShort != 0 || out[0].DeltaLong != 0 || out[0].RollingVar != 0 {
			t.Errorf("%s: expected all derived channels == 0", name)
		}
	}
}

func TestWarmupZero(t *testing.T) {
	p := NewProcessor(60, alpha)
	s := model.RawSample{EntityID: "cpu0", MetricName: "cpu_usage_total_percent", Value: 50}
	out := p.Process([]model.RawSample{s})
	if out[0].DeltaShort != 0 || out[0].DeltaLong != 0 || out[0].RollingVar != 0 {
		t.Error("first sample: expected all derived channels == 0 during warm-up")
	}
}

func TestDeltaShort(t *testing.T) {
	p := NewProcessor(60, alpha)
	id := "cpu0"
	name := "cpu_usage_total_percent"
	vals := []float64{10, 20, 15}
	var outs []model.ProcessedSample
	for _, v := range vals {
		out := p.Process([]model.RawSample{{EntityID: id, MetricName: name, Value: v}})
		outs = append(outs, out...)
	}
	// delta_short[0] = 0 (warm-up)
	// delta_short[1] = 20 - 10 = 10
	// delta_short[2] = 15 - 20 = -5
	if outs[0].DeltaShort != 0 {
		t.Errorf("delta_short[0]: got %v, want 0", outs[0].DeltaShort)
	}
	if outs[1].DeltaShort != 10 {
		t.Errorf("delta_short[1]: got %v, want 10", outs[1].DeltaShort)
	}
	if outs[2].DeltaShort != -5 {
		t.Errorf("delta_short[2]: got %v, want -5", outs[2].DeltaShort)
	}
}

func TestEMAAlpha(t *testing.T) {
	p := NewProcessor(60, alpha)
	id := "n0"
	name := "ram_used_percent"
	// Feed a constant value: EMA should converge to that value.
	const val = 80.0
	for range 200 {
		p.Process([]model.RawSample{{EntityID: id, MetricName: name, Value: val}})
	}
	out := p.Process([]model.RawSample{{EntityID: id, MetricName: name, Value: val}})
	// After convergence, delta_long = value - EMA ≈ 0
	if math.Abs(out[0].DeltaLong) > 0.01 {
		t.Errorf("EMA convergence: delta_long=%v, want ≈0", out[0].DeltaLong)
	}
}

func TestRollingVar(t *testing.T) {
	p := NewProcessor(4, alpha) // small window for test
	id := "h0"
	name := "cpu_temperature_celsius"
	vals := []float64{10, 20, 30, 40}
	for _, v := range vals {
		p.Process([]model.RawSample{{EntityID: id, MetricName: name, Value: v}})
	}
	out := p.Process([]model.RawSample{{EntityID: id, MetricName: name, Value: 50}})
	// Window after 5th push (cap=4): [20,30,40,50]
	// mean = 35, var = ((20-35)²+(30-35)²+(40-35)²+(50-35)²)/4 = (225+25+25+225)/4 = 125
	if math.Abs(out[0].RollingVar-125) > 0.001 {
		t.Errorf("rolling_var: got %v, want 125", out[0].RollingVar)
	}
}

func TestRingBufferVariance(t *testing.T) {
	rb := newRingBuffer(3)
	rb.Push(2)
	rb.Push(4)
	rb.Push(6)
	// mean=4, var=((2-4)²+(4-4)²+(6-4)²)/3 = (4+0+4)/3 = 8/3
	want := 8.0 / 3.0
	got := rb.Variance()
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("Variance: got %v, want %v", got, want)
	}
}
