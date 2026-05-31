//go:build linux

package collector

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// writeFile creates parent dirs and writes content (helper for fake sysfs tree).
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func collectPMem(t *testing.T, ndBase string) []sampleView {
	t.Helper()
	c := &PMemCollector{nodeID: "h04-017", ndBase: ndBase}
	samples, err := c.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	out := make([]sampleView, 0, len(samples))
	for _, s := range samples {
		out = append(out, sampleView{s.EntityType, s.EntityID, s.MetricName, s.Value})
	}
	return out
}

type sampleView struct {
	etype string
	eid   string
	name  string
	val   float64
}

func find(views []sampleView, name string) (sampleView, bool) {
	for _, v := range views {
		if v.name == name {
			return v, true
		}
	}
	return sampleView{}, false
}

func TestPMem_NoDevice_NoSamplesNoError(t *testing.T) {
	views := collectPMem(t, filepath.Join(t.TempDir(), "does-not-exist"))
	if len(views) != 0 {
		t.Fatalf("expected no samples on a host without PMEM, got %d", len(views))
	}
}

func TestPMem_HealthyAndDegraded(t *testing.T) {
	base := t.TempDir()
	// nmem0 healthy (empty flags), nmem1 degraded (a fault flag present).
	writeFile(t, filepath.Join(base, "nmem0", "nfit", "flags"), "\n")
	writeFile(t, filepath.Join(base, "nmem1", "nfit", "flags"), "smart_event\n")
	// One fsdax namespace → mode flag 2.
	writeFile(t, filepath.Join(base, "namespace0.0", "mode"), "fsdax\n")

	views := collectPMem(t, base)

	if len(views) != 4 {
		t.Fatalf("expected 4 pmem metrics, got %d: %+v", len(views), views)
	}
	for _, v := range views {
		if v.etype != "ram" {
			t.Errorf("pmem must fold into the ram node, got entity_type=%q", v.etype)
		}
		if v.eid != "h04-017:ram0" {
			t.Errorf("unexpected entity_id %q", v.eid)
		}
	}
	if h, ok := find(views, "pmem_health_status"); !ok || h.val != 1.0 {
		t.Errorf("expected degraded health=1, got %+v ok=%v", h, ok)
	}
	if m, ok := find(views, "pmem_mode_flag"); !ok || m.val != 2.0 {
		t.Errorf("expected fsdax mode flag=2, got %+v ok=%v", m, ok)
	}
}

func TestPMem_AllHealthy(t *testing.T) {
	base := t.TempDir()
	writeFile(t, filepath.Join(base, "nmem0", "nfit", "flags"), "")
	writeFile(t, filepath.Join(base, "namespace0.0", "mode"), "devdax\n")

	views := collectPMem(t, base)
	if h, ok := find(views, "pmem_health_status"); !ok || h.val != 0.0 {
		t.Errorf("expected healthy health=0, got %+v ok=%v", h, ok)
	}
	if m, ok := find(views, "pmem_mode_flag"); !ok || m.val != 3.0 {
		t.Errorf("expected devdax mode flag=3, got %+v ok=%v", m, ok)
	}
}

func TestPMemModeFlag(t *testing.T) {
	cases := map[string]float64{
		"raw": 0, "sector": 1, "safe": 1, "fsdax": 2,
		"memory": 2, "devdax": 3, "dax": 3, "weird": 0,
	}
	for in, want := range cases {
		if got := pmemModeFlag(in); got != want {
			t.Errorf("pmemModeFlag(%q)=%v want %v", in, got, want)
		}
	}
}
