package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	NodeID         string        `yaml:"node_id"`
	SampleInterval time.Duration `yaml:"sample_interval"`
	Feature        FeatureCfg    `yaml:"feature"`
	Kafka          KafkaCfg      `yaml:"kafka"`
	Collectors     CollectorsCfg `yaml:"collectors"`
}

type FeatureCfg struct {
	LongWindow int     `yaml:"long_window"`
	EmaAlpha   float64 `yaml:"ema_alpha"`
}

type KafkaCfg struct {
	Brokers      []string `yaml:"brokers"`
	Topic        string   `yaml:"topic"`
	PartitionKey string   `yaml:"partition_key"`
}

type CollectorsCfg struct {
	LinuxCPU     CollectorToggle `yaml:"linux_cpu"`
	LinuxMem     CollectorToggle `yaml:"linux_mem"`
	LinuxDisk    CollectorToggle `yaml:"linux_disk"`
	CgroupsSLURM CollectorToggle `yaml:"cgroups_slurm"`
	NvmlGPU      CollectorToggle `yaml:"nvml_gpu"`
	Fabric       FabricCfg       `yaml:"fabric"`
	PMem         CollectorToggle `yaml:"pmem"`
}

type CollectorToggle struct {
	Enabled bool `yaml:"enabled"`
}

type FabricCfg struct {
	Enabled bool    `yaml:"enabled"`
	SNMP    SNMPCfg `yaml:"snmp"`
}

type SNMPCfg struct {
	Targets   []string `yaml:"targets"`
	Community string   `yaml:"community"`
}

func Load(path string) (*Config, error) {
	cfg := defaults()

	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read config %q: %w", path, err)
		}
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse config: %w", err)
		}
	}

	applyEnv(cfg)

	if cfg.NodeID == "" {
		hostname, _ := os.Hostname()
		cfg.NodeID = hostname
	}

	return cfg, validate(cfg)
}

func defaults() *Config {
	return &Config{
		SampleInterval: 5 * time.Second,
		Feature: FeatureCfg{
			LongWindow: 60,
			EmaAlpha:   0.0645,
		},
		Kafka: KafkaCfg{
			Brokers:      []string{"localhost:9092"},
			Topic:        "telemetry.raw",
			PartitionKey: "entity_id",
		},
		Collectors: CollectorsCfg{
			LinuxCPU:     CollectorToggle{Enabled: true},
			LinuxMem:     CollectorToggle{Enabled: true},
			LinuxDisk:    CollectorToggle{Enabled: true},
			CgroupsSLURM: CollectorToggle{Enabled: true},
			// PMEM is safe cluster-wide: nodes without Optane emit nothing.
			PMem: CollectorToggle{Enabled: true},
		},
	}
}

func applyEnv(cfg *Config) {
	if v := os.Getenv("NODE_ID"); v != "" {
		cfg.NodeID = v
	}
	if v := os.Getenv("KAFKA_BROKERS"); v != "" {
		cfg.Kafka.Brokers = strings.Split(v, ",")
	}
	if v := os.Getenv("KAFKA_TOPIC"); v != "" {
		cfg.Kafka.Topic = v
	}
}

func validate(cfg *Config) error {
	if cfg.SampleInterval <= 0 {
		return fmt.Errorf("sample_interval must be positive")
	}
	if len(cfg.Kafka.Brokers) == 0 {
		return fmt.Errorf("kafka.brokers must not be empty")
	}
	return nil
}
