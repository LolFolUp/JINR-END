package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/aladed/JINR/edge-agent/internal/config"
	"github.com/aladed/JINR/edge-agent/internal/collector"
	"github.com/aladed/JINR/edge-agent/internal/pipeline"
	"github.com/aladed/JINR/edge-agent/internal/transport"
)

func main() {
	cfg, err := config.Load(configPath())
	if err != nil {
		slog.Error("load config", "err", err)
		os.Exit(1)
	}

	slog.Info("edge-agent starting", "node_id", cfg.NodeID, "interval", cfg.SampleInterval)

	reg := collector.NewRegistry()
	registerCollectors(reg, cfg)

	producer := transport.NewKafkaProducer(cfg.NodeID, cfg.Kafka)
	defer func() {
		if err := producer.Close(); err != nil {
			slog.Error("kafka close", "err", err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		slog.Info("shutdown signal received")
		cancel()
	}()

	pipeline.New(cfg, reg.All(), producer).Run(ctx)
	slog.Info("edge-agent stopped")
}

func configPath() string {
	if len(os.Args) > 1 {
		return os.Args[1]
	}
	for _, p := range []string{"deploy/agent.example.yaml", "agent.yaml"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
