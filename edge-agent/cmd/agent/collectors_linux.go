//go:build linux

package main

import (
	"github.com/aladed/JINR/edge-agent/internal/collector"
	"github.com/aladed/JINR/edge-agent/internal/config"
)

func registerCollectors(reg *collector.Registry, cfg *config.Config) {
	if cfg.Collectors.LinuxCPU.Enabled {
		reg.Register(collector.NewLinuxCPUCollector(cfg.NodeID))
	}
	if cfg.Collectors.LinuxMem.Enabled {
		reg.Register(collector.NewLinuxMemCollector(cfg.NodeID))
	}
	if cfg.Collectors.LinuxDisk.Enabled {
		reg.Register(collector.NewLinuxDiskCollector(cfg.NodeID))
	}
	if cfg.Collectors.CgroupsSLURM.Enabled {
		reg.Register(collector.NewCgroupsSLURMCollector(cfg.NodeID))
	}
	if cfg.Collectors.Fabric.Enabled {
		reg.Register(collector.NewFabricCollector(cfg.NodeID, cfg.Collectors.Fabric))
	}
	if cfg.Collectors.NvmlGPU.Enabled {
		reg.Register(collector.NewGPUCollector(cfg.NodeID))
	}
	if cfg.Collectors.PMem.Enabled {
		reg.Register(collector.NewPMemCollector(cfg.NodeID))
	}
}
