package model

import "time"

type RawSample struct {
	Timestamp  time.Time
	SourceType string
	EntityType string            // cpu|gpu|hdd|ram|switch|job|link
	EntityID   string            // stable ID, e.g. "h04-017:gpu0", "leaf-01:port12"
	MetricName string            // exactly from §1.2 contract
	Value      float64
	Unit       string
	Labels     map[string]string // job_id, rack, gpu_index, switch_role, link_type...
}

type ProcessedSample struct {
	RawSample
	IsCategorical bool
	DeltaShort    float64
	DeltaLong     float64
	RollingVar    float64
}
