package transport

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/config"
	"github.com/aladed/JINR/edge-agent/internal/model"
	kafka "github.com/segmentio/kafka-go"
)

type KafkaProducer struct {
	writer *kafka.Writer
	cfg    config.KafkaCfg
	nodeID string
}

func NewKafkaProducer(nodeID string, cfg config.KafkaCfg) *KafkaProducer {
	w := &kafka.Writer{
		Addr:         kafka.TCP(cfg.Brokers...),
		Topic:        cfg.Topic,
		Balancer:     &kafka.Hash{},
		BatchTimeout: 10 * time.Millisecond,
		WriteTimeout: 5 * time.Second,
		ReadTimeout:  5 * time.Second,
		MaxAttempts:  3,
	}
	return &KafkaProducer{writer: w, cfg: cfg, nodeID: nodeID}
}

// Send serializes the samples and writes them to Kafka. Errors are logged but
// do not crash the agent.
func (p *KafkaProducer) Send(ctx context.Context, samples []model.ProcessedSample) {
	if len(samples) == 0 {
		return
	}

	payload, err := Serialize(p.nodeID, samples)
	if err != nil {
		slog.Error("serialize batch", "err", err)
		return
	}

	key := p.partitionKey(samples[0])
	msg := kafka.Message{
		Key:   []byte(key),
		Value: payload,
	}

	if err := p.writer.WriteMessages(ctx, msg); err != nil {
		slog.Error("kafka write", "topic", p.cfg.Topic, "key", key, "err", err)
		return
	}

	slog.Debug("kafka batch sent", "topic", p.cfg.Topic, "samples", len(samples), "bytes", len(payload))
}

func (p *KafkaProducer) Close() error {
	return p.writer.Close()
}

func (p *KafkaProducer) partitionKey(s model.ProcessedSample) string {
	switch p.cfg.PartitionKey {
	case "entity_type_entity_id":
		return fmt.Sprintf("%s:%s", s.EntityType, s.EntityID)
	default: // "entity_id"
		return s.EntityID
	}
}
