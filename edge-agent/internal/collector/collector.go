package collector

import (
	"context"
	"fmt"

	"github.com/aladed/JINR/edge-agent/internal/model"
)

type Collector interface {
	Name() string
	SourceType() string
	Collect(ctx context.Context) ([]model.RawSample, error)
}

type Registry struct {
	collectors []Collector
}

func NewRegistry() *Registry {
	return &Registry{}
}

func (r *Registry) Register(c Collector) {
	r.collectors = append(r.collectors, c)
}

func (r *Registry) All() []Collector {
	return r.collectors
}

func (r *Registry) Get(name string) (Collector, error) {
	for _, c := range r.collectors {
		if c.Name() == name {
			return c, nil
		}
	}
	return nil, fmt.Errorf("collector %q not found", name)
}
