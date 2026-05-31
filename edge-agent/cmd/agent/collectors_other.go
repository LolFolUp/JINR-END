//go:build !linux

package main

import (
	"github.com/aladed/JINR/edge-agent/internal/collector"
	"github.com/aladed/JINR/edge-agent/internal/config"
)

// On non-Linux platforms (Windows dev, macOS CI) use the fake collector so the
// agent still starts and exercises the full pipeline.
func registerCollectors(reg *collector.Registry, cfg *config.Config) {
	reg.Register(collector.NewFakeCollector(cfg.NodeID))
}
