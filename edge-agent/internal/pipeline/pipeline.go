package pipeline

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/collector"
	"github.com/aladed/JINR/edge-agent/internal/config"
	"github.com/aladed/JINR/edge-agent/internal/feature"
	"github.com/aladed/JINR/edge-agent/internal/model"
)

// Sender is anything that can ship a batch of processed samples (Kafka, stdout, …).
type Sender interface {
	Send(ctx context.Context, samples []model.ProcessedSample)
}

type Pipeline struct {
	cfg        *config.Config
	collectors []collector.Collector
	processor  *feature.Processor
	sender     Sender // nil → log-only mode
}

func New(cfg *config.Config, collectors []collector.Collector, sender Sender) *Pipeline {
	return &Pipeline{
		cfg:        cfg,
		collectors: collectors,
		processor:  feature.NewProcessor(cfg.Feature.LongWindow, cfg.Feature.EmaAlpha),
		sender:     sender,
	}
}

func (p *Pipeline) Run(ctx context.Context) {
	ticker := time.NewTicker(p.cfg.SampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.tick(ctx)
		}
	}
}

func (p *Pipeline) tick(ctx context.Context) {
	raw := p.collectAll(ctx)
	if len(raw) == 0 {
		return
	}

	processed := p.processor.Process(raw)

	for _, s := range processed {
		slog.Info("sample",
			"entity_type", s.EntityType,
			"entity_id", s.EntityID,
			"metric", s.MetricName,
			"value", s.Value,
		)
	}

	if p.sender != nil {
		p.sender.Send(ctx, processed)
	}
}

func (p *Pipeline) collectAll(ctx context.Context) []model.RawSample {
	type result struct {
		samples []model.RawSample
		err     error
		name    string
	}

	ch := make(chan result, len(p.collectors))
	var wg sync.WaitGroup

	for _, c := range p.collectors {
		wg.Add(1)
		go func(c collector.Collector) {
			defer wg.Done()
			samples, err := c.Collect(ctx)
			ch <- result{samples: samples, err: err, name: c.Name()}
		}(c)
	}

	wg.Wait()
	close(ch)

	var all []model.RawSample
	for r := range ch {
		if r.err != nil {
			slog.Error("collector error", "collector", r.name, "err", r.err)
			continue
		}
		all = append(all, r.samples...)
	}
	return all
}
