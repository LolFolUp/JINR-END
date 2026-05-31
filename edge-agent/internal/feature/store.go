package feature

import "sync"

// metricState holds per-(entity_id+metric_name) running state.
type metricState struct {
	prev  float64
	ema   float64
	buf   *RingBuffer
	count int // total samples seen for this key
}

// Store holds EMA/prev state for every metric key in the node.
// Thread-safe.
type Store struct {
	mu         sync.Mutex
	states     map[string]*metricState
	longWindow int
	emaAlpha   float64
}

func NewStore(longWindow int, emaAlpha float64) *Store {
	return &Store{
		states:     make(map[string]*metricState),
		longWindow: longWindow,
		emaAlpha:   emaAlpha,
	}
}

func (s *Store) get(key string) *metricState {
	st, ok := s.states[key]
	if !ok {
		st = &metricState{buf: newRingBuffer(s.longWindow)}
		s.states[key] = st
	}
	return st
}

// Update advances the state for key with value v and returns
// (deltaShort, deltaLong, rollingVar).
// All outputs are 0 until at least 2 samples have been seen (warm-up).
func (s *Store) Update(key string, value float64) (deltaShort, deltaLong, rollingVar float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	st := s.get(key)
	st.buf.Push(value)
	st.count++

	if st.count == 1 {
		// First sample — initialise EMA and prev; all derived = 0.
		st.ema = value
		st.prev = value
		return 0, 0, 0
	}

	deltaShort = value - st.prev
	st.ema = s.emaAlpha*value + (1-s.emaAlpha)*st.ema
	deltaLong = value - st.ema
	rollingVar = st.buf.Variance()

	st.prev = value
	return deltaShort, deltaLong, rollingVar
}
