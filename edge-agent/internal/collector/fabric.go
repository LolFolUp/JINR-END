//go:build linux

package collector

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aladed/JINR/edge-agent/internal/config"
	"github.com/aladed/JINR/edge-agent/internal/model"
	"github.com/gosnmp/gosnmp"
)

// FabricCollector collects:
//   - Switch metrics from SNMP targets (IF-MIB) → entity_type="switch"
//   - Link metrics from InfiniBand/OmniPath sysfs counters → entity_type="link"
type FabricCollector struct {
	nodeID string
	cfg    config.FabricCfg

	mu       sync.Mutex
	prevSwitch map[string]switchIfStat // key: target+ifIndex
	prevIB     map[string]ibPortStat   // key: dev+port
	prevTs     time.Time
}

func NewFabricCollector(nodeID string, cfg config.FabricCfg) *FabricCollector {
	return &FabricCollector{
		nodeID:     nodeID,
		cfg:        cfg,
		prevSwitch: make(map[string]switchIfStat),
		prevIB:     make(map[string]ibPortStat),
	}
}

func (c *FabricCollector) Name() string       { return "fabric" }
func (c *FabricCollector) SourceType() string { return "fabric" }

// ─────────────────────────────────────────────
// SNMP / Switch metrics
// ─────────────────────────────────────────────

type switchIfStat struct {
	inOctets, outOctets     uint64
	inErrors, inDiscards    uint64
	inPkts, outPkts         uint64
	speedMbps               uint64
	ts                      time.Time
}

// IF-MIB OID prefixes (snmpwalk returns individual indexes).
const (
	oidIfInOctets   = "1.3.6.1.2.1.2.2.1.10"
	oidIfOutOctets  = "1.3.6.1.2.1.2.2.1.16"
	oidIfInErrors   = "1.3.6.1.2.1.2.2.1.14"
	oidIfInDiscards = "1.3.6.1.2.1.2.2.1.13"
	oidIfInPkts     = "1.3.6.1.2.1.2.2.1.11"
	oidIfOutPkts    = "1.3.6.1.2.1.2.2.1.17"
	oidIfHighSpeed  = "1.3.6.1.2.1.31.1.1.1.15" // Mbps
	oidIfOperStatus = "1.3.6.1.2.1.2.2.1.8"
)

func snmpWalkTable(target, community string, oid string) (map[string]uint64, error) {
	g := &gosnmp.GoSNMP{
		Target:    target,
		Port:      161,
		Community: community,
		Version:   gosnmp.Version2c,
		Timeout:   2 * time.Second,
		Retries:   1,
	}
	if err := g.Connect(); err != nil {
		return nil, fmt.Errorf("snmp connect %s: %w", target, err)
	}
	defer g.Conn.Close()

	result := make(map[string]uint64)
	err := g.Walk(oid, func(pdu gosnmp.SnmpPDU) error {
		// PDU name is full OID; extract the index (last element).
		parts := strings.Split(pdu.Name, ".")
		idx := parts[len(parts)-1]
		switch pdu.Type {
		case gosnmp.Counter32, gosnmp.Gauge32, gosnmp.TimeTicks, gosnmp.Integer:
			result[idx] = uint64(gosnmp.ToBigInt(pdu.Value).Int64())
		case gosnmp.Counter64:
			result[idx] = gosnmp.ToBigInt(pdu.Value).Uint64()
		}
		return nil
	})
	return result, err
}

func (c *FabricCollector) collectSwitch(now time.Time, prevTs time.Time) ([]model.RawSample, map[string]switchIfStat) {
	newPrev := make(map[string]switchIfStat)
	var samples []model.RawSample

	for _, target := range c.cfg.SNMP.Targets {
		inOctets, err := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfInOctets)
		if err != nil {
			continue
		}
		outOctets, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfOutOctets)
		inErrors, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfInErrors)
		inDiscards, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfInDiscards)
		inPkts, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfInPkts)
		outPkts, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfOutPkts)
		highSpeed, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfHighSpeed)
		operStatus, _ := snmpWalkTable(target, c.cfg.SNMP.Community, oidIfOperStatus)

		for idx := range inOctets {
			cur := switchIfStat{
				inOctets:  inOctets[idx],
				outOctets: outOctets[idx],
				inErrors:  inErrors[idx],
				inDiscards: inDiscards[idx],
				inPkts:    inPkts[idx],
				outPkts:   outPkts[idx],
				speedMbps: highSpeed[idx],
				ts:        now,
			}
			key := target + ":" + idx
			newPrev[key] = cur

			ps, hasPrev := c.prevSwitch[key]
			dt := now.Sub(prevTs).Seconds()
			if !hasPrev || prevTs.IsZero() || dt <= 0 {
				continue
			}

			const byteToMB = 1.0 / 1024 / 1024
			rxMBs := float64(cur.inOctets-ps.inOctets) * byteToMB / dt
			txMBs := float64(cur.outOctets-ps.outOctets) * byteToMB / dt

			speedMbps := float64(cur.speedMbps)
			if speedMbps <= 0 {
				speedMbps = 1000
			}
			bwUsedPct := (rxMBs + txMBs) * 8 / speedMbps * 100
			pktLoss := 0.0
			totalPkts := float64(cur.inPkts+cur.outPkts - ps.inPkts - ps.outPkts)
			if totalPkts > 0 {
				pktLoss = float64(cur.inErrors+cur.inDiscards-ps.inErrors-ps.inDiscards) / totalPkts * 100
			}
			crcErrors := float64(cur.inErrors - ps.inErrors)
			status := operStatus[idx]

			eid := target + ":port" + idx
			labels := map[string]string{"switch_role": "leaf"}
			add := func(name string, val float64, unit string) {
				s := metric(now, "fabric", "switch", eid, name, val, unit)
				s.Labels = labels
				samples = append(samples, s)
			}

			add("switch_port_utilization_percent", bwUsedPct, "percent")
			add("switch_packet_loss_percent", pktLoss, "percent")
			add("switch_crc_errors_count", crcErrors, "count")
			add("switch_latency_ms", 0, "ms")
			add("switch_bandwidth_usage_percent", bwUsedPct, "percent")
			add("switch_temperature_celsius", 0, "celsius")
			add("switch_power_watts", 0, "watts")
			add("switch_active_connections_count", float64(totalPkts), "count")
			add("switch_rx_mb", rxMBs, "mb")
			add("switch_tx_mb", txMBs, "mb")
			_ = status
		}
	}
	return samples, newPrev
}

// ─────────────────────────────────────────────
// InfiniBand / OmniPath sysfs → link metrics
// ─────────────────────────────────────────────

type ibPortStat struct {
	rcvData, xmitData     uint64
	rcvErrors, xmitDisc   uint64
	rcvPkts, xmitPkts     uint64
	linkDowned            uint64
	symbolErrors          uint64
	ts                    time.Time
}

const ibBase = "/sys/class/infiniband"

func ibUint64(path string) uint64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	return v
}

func (c *FabricCollector) collectIB(now time.Time, prevTs time.Time) ([]model.RawSample, map[string]ibPortStat) {
	newPrev := make(map[string]ibPortStat)
	var samples []model.RawSample

	devs, _ := filepath.Glob(ibBase + "/*/ports/*/counters")
	for _, counterDir := range devs {
		// counterDir: /sys/class/infiniband/mlx5_0/ports/1/counters
		parts := strings.Split(counterDir, "/")
		if len(parts) < 6 {
			continue
		}
		devName := parts[4]
		portNum := parts[6]
		key := devName + ":" + portNum

		cur := ibPortStat{
			rcvData:      ibUint64(filepath.Join(counterDir, "port_rcv_data")),
			xmitData:     ibUint64(filepath.Join(counterDir, "port_xmit_data")),
			rcvErrors:    ibUint64(filepath.Join(counterDir, "port_rcv_errors")),
			xmitDisc:     ibUint64(filepath.Join(counterDir, "port_xmit_discards")),
			rcvPkts:      ibUint64(filepath.Join(counterDir, "port_rcv_packets")),
			xmitPkts:     ibUint64(filepath.Join(counterDir, "port_xmit_packets")),
			linkDowned:   ibUint64(filepath.Join(counterDir, "link_downed")),
			symbolErrors: ibUint64(filepath.Join(counterDir, "symbol_error")),
			ts:           now,
		}
		newPrev[key] = cur

		ps, hasPrev := c.prevIB[key]
		dt := now.Sub(prevTs).Seconds()
		if !hasPrev || prevTs.IsZero() || dt <= 0 {
			continue
		}

		// IB port_rcv_data is in 32-byte words (IB spec).
		const ibWordBytes = 4.0
		const byteToMB = 1.0 / 1024 / 1024
		rxMBs := float64(cur.rcvData-ps.rcvData) * ibWordBytes * byteToMB / dt
		txMBs := float64(cur.xmitData-ps.xmitData) * ibWordBytes * byteToMB / dt

		totalPkts := float64(cur.rcvPkts + cur.xmitPkts - ps.rcvPkts - ps.xmitPkts)
		errPkts := float64(cur.rcvErrors + cur.xmitDisc - ps.rcvErrors - ps.xmitDisc)
		pktLoss := 0.0
		if totalPkts > 0 {
			pktLoss = errPkts / totalPkts * 100
		}
		errRate := 0.0
		if totalPkts > 0 {
			errRate = errPkts / totalPkts
		}

		// Assume 100 Gbps IB link for utilisation estimate.
		const linkSpeedMbps = 100_000.0
		utilRatio := (rxMBs + txMBs) * 8 / linkSpeedMbps

		eid := c.nodeID + ":" + devName + ":p" + portNum
		labels := map[string]string{"link_type": "infiniband"}
		add := func(name string, val float64, unit string) {
			s := metric(now, "fabric", "link", eid, name, val, unit)
			s.Labels = labels
			samples = append(samples, s)
		}

		linkStatus := 1.0 // assume up
		if cur.linkDowned > ps.linkDowned {
			linkStatus = 0
		}

		add("link_bandwidth_usage_percent", utilRatio*100, "percent")
		add("link_latency_ms", 0, "ms")
		add("link_packet_loss_percent", pktLoss, "percent")
		add("link_jitter_ms", 0, "ms")
		add("link_crc_errors_count", float64(cur.symbolErrors-ps.symbolErrors), "count")
		add("link_rx_mb", rxMBs, "mb")
		add("link_tx_mb", txMBs, "mb")
		add("link_dropped_packets_count", float64(cur.xmitDisc-ps.xmitDisc), "count")
		add("link_retransmits_count", 0, "count")
		add("link_queue_depth_count", 0, "count")
		add("link_signal_strength_dbm", 0, "dbm")
		add("link_power_watts", 0, "watts")
		add("link_utilization_ratio", utilRatio, "ratio")
		add("link_congestion_score", pktLoss/100, "ratio")
		add("link_timeout_count", 0, "count")
		add("link_reset_count", float64(cur.linkDowned-ps.linkDowned), "count")
		add("link_availability_percent", linkStatus*100, "percent")
		add("link_error_rate", errRate, "ratio")
		add("link_status_encoded", linkStatus, "encoded")
	}
	return samples, newPrev
}

func (c *FabricCollector) Collect(_ context.Context) ([]model.RawSample, error) {
	now := time.Now()

	c.mu.Lock()
	prevSwitch := c.prevSwitch
	prevIB := c.prevIB
	prevTs := c.prevTs
	c.mu.Unlock()

	swSamples, newSw := c.collectSwitch(now, prevTs)
	ibSamples, newIB := c.collectIB(now, prevTs)

	c.mu.Lock()
	for k, v := range newSw {
		prevSwitch[k] = v
	}
	for k, v := range newIB {
		prevIB[k] = v
	}
	c.prevSwitch = prevSwitch
	c.prevIB = prevIB
	c.prevTs = now
	c.mu.Unlock()

	return append(swSamples, ibSamples...), nil
}
