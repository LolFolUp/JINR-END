"""Redis-backed technical context store for root-cause nodes."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple


NODE_CONTEXTS: Dict[str, Dict[str, Any]] = {
    "S2": {
        "node_id": "S2",
        "node_type": "switch",
        "hostname": "leaf-switch-02.govorun.jinr.ru",
        "os": "Cumulus Linux 4.4",
        "location": "Rack-B, Slot-3",
        "connected_hosts": ["host-041", "host-042", "host-043", "host-044"],
        "running_services": ["BGP daemon", "SNMP agent", "sFlow collector"],
        "last_maintenance": "2026-03-15",
        "sla_tier": "CRITICAL",
    },
    "HDD-7": {
        "node_id": "HDD-7",
        "node_type": "hdd",
        "hostname": "storage-ost-07.govorun.jinr.ru",
        "os": "Rocky Linux 9.3",
        "disk_model": "Seagate Exos X18",
        "capacity_tb": 18,
        "smart_status": "DEGRADED",
        "mount_points": ["/mnt/lustre/ost07", "/var/lib/checkpoints"],
        "last_maintenance": "2026-02-02",
        "sla_tier": "HIGH",
    },
    "RAM-3": {
        "node_id": "RAM-3",
        "node_type": "ram",
        "hostname": "compute-003.govorun.jinr.ru",
        "os": "Rocky Linux 9.3",
        "total_gb": 512,
        "numa_domains": 4,
        "memory_type": "DDR5 ECC",
        "running_services": ["slurmd", "node_exporter", "dcgm-exporter"],
        "last_maintenance": "2026-04-08",
        "sla_tier": "HIGH",
    },
}


class RedisContextStore:
    """Fetch technical node context from Redis, fakeredis, or a dict fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", client: Optional[Any] = None) -> None:
        self._dict_store = deepcopy(NODE_CONTEXTS)
        self._client = client
        self.context_source = "dict_fallback"

        if self._client is not None:
            self.context_source = "redis"
            self._seed_redis()
            return

        self._client = self._connect_redis(redis_url) or self._connect_fakeredis()
        if self._client is not None:
            self.context_source = "redis"
            self._seed_redis()

    def _connect_redis(self, redis_url: str) -> Optional[Any]:
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=0.1,
                socket_timeout=0.1,
            )
            client.ping()
            return client
        except Exception:
            return None

    def _connect_fakeredis(self) -> Optional[Any]:
        try:
            import fakeredis

            return fakeredis.FakeRedis(decode_responses=True)
        except Exception:
            return None

    def _seed_redis(self) -> None:
        if self._client is None:
            return
        for node_id, context in self._dict_store.items():
            key = self._key(node_id)
            try:
                if not self._client.exists(key):
                    self._client.set(key, json.dumps(context))
            except Exception:
                self._client = None
                self.context_source = "dict_fallback"
                return

    @staticmethod
    def _key(node_id: str) -> str:
        return f"node_context:{node_id}"

    def fetch_context(self, rc_node: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Return context and source label for a GNN root-cause node."""
        node_id = str(rc_node.get("id", "unknown"))
        context = self._fetch_from_redis(node_id) or self._dict_store.get(node_id)
        if context is None:
            context = self._default_context(rc_node)
        return deepcopy(context), self.context_source

    def _fetch_from_redis(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self._client is None:
            return None
        try:
            raw = self._client.get(self._key(node_id))
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            self._client = None
            self.context_source = "dict_fallback"
        return None

    def _default_context(self, rc_node: Dict[str, Any]) -> Dict[str, Any]:
        node_id = str(rc_node.get("id", "unknown"))
        node_type = str(rc_node.get("type", "unknown"))
        hostname_prefix = {
            "switch": "leaf-switch",
            "hdd": "storage-node",
            "ram": "compute-node",
        }.get(node_type, "cluster-node")
        return {
            "node_id": node_id,
            "node_type": node_type,
            "hostname": f"{hostname_prefix}-{node_id.lower()}.govorun.jinr.ru",
            "os": "unknown",
            "location": "unknown",
            "running_services": [],
            "sla_tier": "UNKNOWN",
        }
