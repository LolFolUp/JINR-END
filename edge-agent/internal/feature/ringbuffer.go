package feature

// RingBuffer is a fixed-capacity circular buffer of float64 values.
// O(1) push, no allocations after construction.
type RingBuffer struct {
	data []float64
	head int // next write index
	size int // current number of valid elements
	cap  int
}

func newRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{data: make([]float64, capacity), cap: capacity}
}

func (r *RingBuffer) Push(v float64) {
	r.data[r.head] = v
	r.head = (r.head + 1) % r.cap
	if r.size < r.cap {
		r.size++
	}
}

// Len returns the number of valid elements currently stored.
func (r *RingBuffer) Len() int { return r.size }

// Mean returns the mean of stored values. Caller must ensure Len() > 0.
func (r *RingBuffer) Mean() float64 {
	sum := 0.0
	for i := range r.size {
		sum += r.data[i]
	}
	return sum / float64(r.size)
}

// Variance returns the population variance of stored values.
// Returns 0 if fewer than 2 values are stored.
func (r *RingBuffer) Variance() float64 {
	if r.size < 2 {
		return 0
	}
	mean := r.Mean()
	sq := 0.0
	for i := range r.size {
		d := r.data[i] - mean
		sq += d * d
	}
	return sq / float64(r.size)
}
