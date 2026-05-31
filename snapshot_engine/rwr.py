"""
rwr.py — Random Walk with Restart for global root-cause ranking (L4).

The GNN's per-node anomaly scores and per-edge attention/salience weights are
*local* signals. RWR aggregates them into a *global* root-cause ranking, exactly
as described in the diploma (section 2.3.4):

    r^(k) = (1 - c) · Ã · r^(k-1) + c · p

  Ã : transition matrix — transpose of the attention/salience matrix,
      L1-normalised per column (column-stochastic).
  p : restart vector — node anomaly scores, normalised to sum 1. The walker
      teleports back to anomalous symptoms with probability c.
  c : restart probability ∈ (0, 1).

A "virtual diagnosis agent" starts at the symptoms (anomalous nodes via p) and
walks the dependency edges; probability mass concentrates on the node that is
both anomalous and strongly connected to other anomalous nodes — the cascade
root. The argmax of the stationary distribution r^(∞) is the root cause.

Inputs are exactly what predict() already produces:
  - edges:          [{source, target, weight, relation}]  (edge_xai)
  - restart_scores: {node_id: anomaly_score}              (from scores_by_type)

Pure Python (no torch / numpy): the XAI edge set is small (<= ~60 edges), so a
plain dict-based power iteration converges in well under a millisecond.

The walker travels from each symptom toward its cause. Because the schema's
edges are not uniformly oriented symptom→cause (connected_to points cpu→switch,
the switch being causal infra; attached_to points hdd→cpu, the hdd component
being causal), the transition direction is set per relation by
_WALK_TOWARD_CAUSE so probability mass always accumulates on the true root —
whether it is a shared hub (switch) or a leaf component (hdd/ram). This is the
concrete realisation of the diploma's "transpose the attention matrix" (2.3.4):
orient transitions effect→cause regardless of the raw edge direction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Classic PageRank/RWR restart probability. Higher → trust local symptoms more;
# lower → let topology propagate further toward the root.
DEFAULT_RESTART_PROB: float = 0.15

# Direction the diagnosis walker travels along each relation to move from a
# symptom toward its cause. "forward" = source→target, "reverse" = target→source.
# Mass leaves the symptom node and accumulates on the causal node.
_WALK_TOWARD_CAUSE: Dict[str, str] = {
    "connected_to": "forward",   # cpu/gpu → switch : shared infra is causal
    "executes_on":  "forward",   # job → cpu        : the resource is causal
    "uplink_to":    "forward",   # leaf → spine     : spine is more central
    "attached_to":  "reverse",   # ram/hdd → cpu    : the component is causal
}
_DEFAULT_DIRECTION: str = "forward"


def random_walk_with_restart(
    edges: List[Dict[str, Any]],
    restart_scores: Dict[str, float],
    *,
    restart_prob: float = DEFAULT_RESTART_PROB,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Run RWR over the weighted edge graph. Returns {node_id: rc_score}.

    rc_score is the stationary probability mass — higher = more likely the
    cascade root. Scores sum to 1 across all nodes present in the graph.
    Transition direction per relation comes from _WALK_TOWARD_CAUSE.
    """
    if not (0.0 < restart_prob < 1.0):
        raise ValueError("restart_prob must be in (0, 1)")

    # ── 1. Node set: every endpoint plus every node with a restart score ──────
    node_set = set(restart_scores.keys())
    for e in edges:
        node_set.add(e["source"])
        node_set.add(e["target"])
    if not node_set:
        return {}

    nodes = sorted(node_set)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # ── 2. Weighted adjacency as sparse out-edge lists ────────────────────────
    # raw[j] = list of (i, w): mass flows from column j to node i. The walker
    # moves symptom → cause, with per-relation direction from _WALK_TOWARD_CAUSE,
    # so mass drains out of symptoms and pools on the causal node.
    raw: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for e in edges:
        w = float(e.get("weight", 0.0))
        if w <= 0.0:
            continue
        s = index[e["source"]]
        t = index[e["target"]]
        direction = _WALK_TOWARD_CAUSE.get(e.get("relation", ""), _DEFAULT_DIRECTION)
        if direction == "reverse":
            raw[t].append((s, w))   # symptom=target → cause=source
        else:
            raw[s].append((t, w))   # symptom=source → cause=target

    # ── 3. Column-stochastic transition matrix Ã (normalise each out-list) ────
    # cols[j] = [(i, prob)] with sum of probs == 1. Dangling columns (no out-
    # edges) are handled in the iteration by teleporting their mass via p.
    cols: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    dangling: List[int] = []
    for j in range(n):
        total = sum(w for _, w in raw[j])
        if total <= 0.0:
            dangling.append(j)
            continue
        cols[j] = [(i, w / total) for i, w in raw[j]]

    # ── 4. Restart vector p (normalised anomaly scores) ───────────────────────
    p = [0.0] * n
    s_total = sum(max(0.0, v) for v in restart_scores.values())
    if s_total > 0.0:
        for node, score in restart_scores.items():
            p[index[node]] = max(0.0, score) / s_total
    else:
        # No anomaly signal → uniform restart.
        p = [1.0 / n] * n

    # ── 5. Power iteration: r = (1-c)·Ã·r + c·p ───────────────────────────────
    c = restart_prob
    r = list(p)  # warm start from the restart distribution
    for _ in range(max_iter):
        r_new = [c * p_i for p_i in p]

        # Dangling mass (columns with no out-edges) redistributes via p so the
        # walk stays a proper distribution (sums to 1).
        dangling_mass = sum(r[j] for j in dangling)
        if dangling_mass > 0.0:
            factor = (1.0 - c) * dangling_mass
            for i in range(n):
                r_new[i] += factor * p[i]

        # Propagate mass along normalised columns.
        for j in range(n):
            rj = r[j]
            if rj == 0.0 or not cols[j]:
                continue
            base = (1.0 - c) * rj
            for i, prob in cols[j]:
                r_new[i] += base * prob

        # L1 convergence check.
        delta = sum(abs(r_new[k] - r[k]) for k in range(n))
        r = r_new
        if delta < tol:
            break

    # Normalise (guards against tiny drift).
    total = sum(r)
    if total > 0.0:
        r = [v / total for v in r]

    return {nodes[i]: r[i] for i in range(n)}


def rank_root_causes(
    edges: List[Dict[str, Any]],
    restart_scores: Dict[str, float],
    *,
    top_k: int = 5,
    restart_prob: float = DEFAULT_RESTART_PROB,
) -> List[Dict[str, Any]]:
    """RWR convenience wrapper → ranked [{id, score}] root-cause candidates."""
    scores = random_walk_with_restart(
        edges, restart_scores, restart_prob=restart_prob
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"id": nid, "score": round(s, 6)} for nid, s in ranked[:top_k]]


def restart_scores_from_edges(
    edges: List[Dict[str, Any]],
    scores_by_type: Dict[str, List[float]],
) -> Dict[str, float]:
    """Build the restart vector for the RWR subgraph.

    Looks up each edge endpoint's anomaly score from scores_by_type by parsing
    the "{type}-{idx}" node id. Endpoints whose score is unavailable are skipped.
    """
    restart: Dict[str, float] = {}
    for e in edges:
        for node_id in (e.get("source", ""), e.get("target", "")):
            if not node_id or node_id in restart:
                continue
            parts = node_id.rsplit("-", 1)
            if len(parts) != 2:
                continue
            ntype, sidx = parts
            vec = scores_by_type.get(ntype)
            if vec is None:
                continue
            try:
                idx = int(sidx)
            except ValueError:
                continue
            if 0 <= idx < len(vec):
                restart[node_id] = float(vec[idx])
    return restart
