package transport

import (
	"time"

	"github.com/aladed/JINR/edge-agent/internal/model"
	"github.com/aladed/JINR/edge-agent/internal/transport/pb"
	"google.golang.org/protobuf/proto"
)

// Serialize encodes a batch of processed samples for one node into protobuf bytes.
func Serialize(nodeID string, samples []model.ProcessedSample) ([]byte, error) {
	batch := &pb.Batch{
		NodeId:               nodeID,
		BatchTimestampUnixNs: time.Now().UnixNano(),
		Samples:              make([]*pb.Sample, 0, len(samples)),
	}

	for _, s := range samples {
		ps := &pb.Sample{
			TimestampUnixNs: s.Timestamp.UnixNano(),
			NodeId:          nodeID,
			SourceType:      s.SourceType,
			EntityType:      s.EntityType,
			EntityId:        s.EntityID,
			MetricName:      s.MetricName,
			Unit:            s.Unit,
			IsCategorical:   s.IsCategorical,
			Value:           s.Value,
			DeltaShort:      s.DeltaShort,
			DeltaLong:       s.DeltaLong,
			RollingVar:      s.RollingVar,
			Labels:          s.Labels,
		}
		batch.Samples = append(batch.Samples, ps)
	}

	return proto.Marshal(batch)
}
