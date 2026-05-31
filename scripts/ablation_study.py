"""Ablation study: что реально даёт GNN vs табличные методы.

Гипотеза: XGBoost 86.7% — не потому что умный, а потому что мы дали ему
уже готовый delta_long — признак, который мы специально разработали чтобы
RC-узел выделялся. Настоящий тест структурного преимущества GNN:

Эксперименты:
1. XGBoost-VALUE   : только значение метрики (value channel), без временного анализа
2. XGBoost-TEMPORAL: полный x-вектор (value+delta_short+delta_long+rolling_var)
3. XGBoost-GRAPH   : полный x-вектор + агрегация признаков соседей из edge_index
4. GATv2Hetero     : GNN, обученный end-to-end на графе

Если наша гипотеза верна:
  VALUE << TEMPORAL — наша feature engineering даёт большой прирост
  TEMPORAL < GRAPH < GNN — структура графа добавляет сверху
"""

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_pipeline.eval_utils import build_train_val_indices, RC_CANDIDATE_TYPES, FAULT_TYPES

RAW_DIR   = Path(r"d:\Vlad\JINR-rag-exp-rca70\dataset\v5a_40\raw")
CKPT_PATH = ROOT / "gnn" / "checkpoints" / "best_v5a_40_screening.pt"
META_PATH = ROOT / "gnn" / "artifacts" / "metadata.json"
DATASET_SIZE = 4500
SEED = 42

_MAX_DIM     = 46   # GPU имеет 46 признаков
_N_CONT      = {    # количество непрерывных признаков по типу
    "cpu": 8, "gpu": 11, "ram": 8, "hdd": 8, "switch": 10
}
_CHANNELS    = 4    # value / delta_short / delta_long / rolling_var


def load_graph(idx):
    return torch.load(RAW_DIR / f"data_{idx}.pt", weights_only=False)


# ── извлечение признаков в разных вариантах ───────────────────────────────────

def get_value_only(graph) -> Tuple[np.ndarray, np.ndarray, int]:
    """Только value-канал (channel 0 каждого признака). Имитирует наивный мониторинг."""
    rows_X, rows_y = [], []
    for ti, nt in enumerate(RC_CANDIDATE_TYPES):
        if nt not in graph.node_types:
            continue
        x = graph[nt].x.float().numpy()   # [N, feat_dim]
        y = graph[nt].y.float().numpy()
        n_cont = _N_CONT.get(nt, x.shape[1] // _CHANNELS)
        # value-канал: индексы 0, 4, 8, ... (каждые 4 колонки)
        value_cols = [i * _CHANNELS for i in range(n_cont) if i * _CHANNELS < x.shape[1]]
        x_val = x[:, value_cols]
        # pad + type
        if x_val.shape[1] < _MAX_DIM // _CHANNELS:
            pad = np.zeros((x_val.shape[0], _MAX_DIM // _CHANNELS - x_val.shape[1]), dtype=np.float32)
            x_val = np.concatenate([x_val, pad], axis=1)
        type_oh = np.zeros((x_val.shape[0], len(RC_CANDIDATE_TYPES)), dtype=np.float32)
        type_oh[:, ti] = 1.0
        rows_X.append(np.concatenate([x_val, type_oh], axis=1))
        rows_y.append(y)
    if not rows_X:
        return np.empty((0, 0)), np.empty(0), -1
    fi = int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1
    return np.concatenate(rows_X), np.concatenate(rows_y), fi


def get_full_temporal(graph) -> Tuple[np.ndarray, np.ndarray, int]:
    """Полный x-вектор (все 4 канала). Это то, что тестировали вчера."""
    rows_X, rows_y = [], []
    for ti, nt in enumerate(RC_CANDIDATE_TYPES):
        if nt not in graph.node_types:
            continue
        x = graph[nt].x.float().numpy()
        y = graph[nt].y.float().numpy()
        if x.shape[1] < _MAX_DIM:
            pad = np.zeros((x.shape[0], _MAX_DIM - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        type_oh = np.zeros((x.shape[0], len(RC_CANDIDATE_TYPES)), dtype=np.float32)
        type_oh[:, ti] = 1.0
        rows_X.append(np.concatenate([x, type_oh], axis=1))
        rows_y.append(y)
    if not rows_X:
        return np.empty((0, 0)), np.empty(0), -1
    fi = int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1
    return np.concatenate(rows_X), np.concatenate(rows_y), fi


def get_graph_features(graph) -> Tuple[np.ndarray, np.ndarray, int]:
    """Полный x-вектор + агрегация признаков СОСЕДЕЙ (mean, max по edge_index).

    Это то, что должен был бы делать GNN вручную.
    Для каждого RC-кандидата добавляем mean/max x-векторов его прямых соседей.
    """
    # Сначала собираем узлы
    node_x: Dict[str, np.ndarray] = {}
    for nt in graph.node_types:
        node_x[nt] = graph[nt].x.float().numpy()

    rows_X, rows_y = [], []
    for ti, nt in enumerate(RC_CANDIDATE_TYPES):
        if nt not in graph.node_types:
            continue
        x = node_x[nt]                    # [N, feat_dim]
        y = graph[nt].y.float().numpy()
        n_nodes = x.shape[0]

        # Агрегируем признаки соседей по исходящим и входящим рёбрам
        # Только физические рёбра (не reports_to)
        neigh_sum  = np.zeros((n_nodes, _MAX_DIM), dtype=np.float32)
        neigh_max  = np.full((n_nodes, _MAX_DIM), -1e6, dtype=np.float32)
        neigh_cnt  = np.zeros(n_nodes, dtype=np.float32)

        for et in graph.edge_types:
            src_t, rel, dst_t = et
            if "reports_to" in rel:
                continue
            ei = graph[et].edge_index.numpy()  # [2, E]

            if src_t == nt:
                # nt → dst_t: узлы dst_t — соседи нашего узла
                dst_x = node_x.get(dst_t)
                if dst_x is None:
                    continue
                dst_dim = dst_x.shape[1]
                for e in range(ei.shape[1]):
                    u, v = int(ei[0, e]), int(ei[1, e])
                    if u >= n_nodes:
                        continue
                    feat = dst_x[v] if v < len(dst_x) else np.zeros(dst_dim)
                    padded = np.zeros(_MAX_DIM, dtype=np.float32)
                    padded[:min(dst_dim, _MAX_DIM)] = feat[:min(dst_dim, _MAX_DIM)]
                    neigh_sum[u] += padded
                    neigh_max[u] = np.maximum(neigh_max[u], padded)
                    neigh_cnt[u] += 1

            if dst_t == nt:
                # src_t → nt: узлы src_t — соседи нашего узла
                src_x = node_x.get(src_t)
                if src_x is None:
                    continue
                src_dim = src_x.shape[1]
                for e in range(ei.shape[1]):
                    u, v = int(ei[0, e]), int(ei[1, e])
                    if v >= n_nodes:
                        continue
                    feat = src_x[u] if u < len(src_x) else np.zeros(src_dim)
                    padded = np.zeros(_MAX_DIM, dtype=np.float32)
                    padded[:min(src_dim, _MAX_DIM)] = feat[:min(src_dim, _MAX_DIM)]
                    neigh_sum[v] += padded
                    neigh_max[v] = np.maximum(neigh_max[v], padded)
                    neigh_cnt[v] += 1

        cnt = neigh_cnt[:, None].clip(min=1)
        neigh_mean = neigh_sum / cnt
        neigh_max[neigh_max < -9e5] = 0.0  # узлы без соседей

        # Собираем: own_x (pad) | neigh_mean | neigh_max | type_oh
        x_pad = np.zeros((n_nodes, _MAX_DIM), dtype=np.float32)
        x_pad[:, :min(x.shape[1], _MAX_DIM)] = x[:, :min(x.shape[1], _MAX_DIM)]
        type_oh = np.zeros((n_nodes, len(RC_CANDIDATE_TYPES)), dtype=np.float32)
        type_oh[:, ti] = 1.0
        x_full = np.concatenate([x_pad, neigh_mean, neigh_max, type_oh], axis=1)
        rows_X.append(x_full)
        rows_y.append(y)

    if not rows_X:
        return np.empty((0, 0)), np.empty(0), -1
    fi = int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1
    return np.concatenate(rows_X), np.concatenate(rows_y), fi


# ── метрики ───────────────────────────────────────────────────────────────────

def eval_sklearn(model, val_graphs, feature_fn) -> Dict:
    ranks, fault_idxs = [], []
    for graph in val_graphs:
        X, y, fi = feature_fn(graph)
        if y.sum() == 0 or X.shape[0] == 0:
            continue
        scores = model.predict_proba(X)[:, 1]
        order = np.argsort(scores)[::-1]
        rc_rank = int(np.where(order == np.argmax(y))[0][0]) + 1
        ranks.append(rc_rank)
        fault_idxs.append(fi)
    return _metrics(ranks, fault_idxs)


def _empty_edges_like(graph) -> Tuple[Dict, Dict]:
    """Build zero-edge dictionaries preserving all edge types and edge dims."""
    ei_d, ea_d = {}, {}
    for et in graph.edge_types:
        ea = graph[et].edge_attr.float()
        ei_d[et] = torch.empty((2, 0), dtype=graph[et].edge_index.dtype)
        ea_d[et] = torch.empty((0, ea.shape[1]), dtype=ea.dtype)
    return ei_d, ea_d


def _random_edges_like(graph, seed: int) -> Tuple[Dict, Dict]:
    """Randomize topology while preserving edge types, counts and edge_attr."""
    rng = torch.Generator().manual_seed(seed)
    ei_d, ea_d = {}, {}
    for et in graph.edge_types:
        src_t, _, dst_t = et
        e = graph[et].edge_index.shape[1]
        src_n = graph[src_t].x.shape[0]
        dst_n = graph[dst_t].x.shape[0]
        src = torch.randint(0, src_n, (e,), generator=rng, dtype=graph[et].edge_index.dtype)
        dst = torch.randint(0, dst_n, (e,), generator=rng, dtype=graph[et].edge_index.dtype)
        ei_d[et] = torch.stack([src, dst], dim=0)
        ea_d[et] = graph[et].edge_attr.float()
    return ei_d, ea_d


def eval_gnn(val_graphs, mode: str = "normal") -> Dict:
    from gnn.model import GATv2Hetero
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    m = GATv2Hetero(node_dims=ckpt["node_dims"], edge_types=ckpt["edge_types"],
                    scorer_mode=ckpt.get("scorer_mode", "shared"))
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()

    ranks, fault_idxs = [], []
    for gi, graph in enumerate(val_graphs):
        x_d  = {nt: graph[nt].x.float() for nt in graph.node_types}
        if mode == "no_edges":
            ei_d, ea_d = _empty_edges_like(graph)
        elif mode == "random_edges":
            ei_d, ea_d = _random_edges_like(graph, seed=10_000 + gi)
        else:
            ei_d = {et: graph[et].edge_index for et in graph.edge_types}
            ea_d = {et: graph[et].edge_attr.float() for et in graph.edge_types}
        with torch.no_grad():
            if mode == "local_only":
                h = {
                    nt: torch.nn.functional.elu(m.input_proj[f"proj_{nt}"](x.float()))
                    for nt, x in x_d.items()
                    if f"proj_{nt}" in m.input_proj
                }
                logits = {}
                for nt in RC_CANDIDATE_TYPES:
                    if nt in h:
                        type_idx = RC_CANDIDATE_TYPES.index(nt)
                        logits[nt] = m.shared_scorer(h[nt], type_idx).squeeze(-1)
            else:
                logits = m(x_d, ei_d, ea_d)
        cands, rc_idx, offset = [], None, 0
        for nt in RC_CANDIDATE_TYPES:
            if nt not in logits:
                continue
            sc = torch.sigmoid(logits[nt]).numpy()
            y  = graph[nt].y.numpy()
            for i in range(len(sc)):
                if y[i] == 1:
                    rc_idx = offset + i
                cands.append((float(sc[i]), offset + i))
            offset += len(sc)
        if rc_idx is None:
            continue
        cands.sort(reverse=True)
        rc_rank = next(r+1 for r, (_, idx) in enumerate(cands) if idx == rc_idx)
        ranks.append(rc_rank)
        fault_idxs.append(int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1)
    return _metrics(ranks, fault_idxs)


def _metrics(ranks, fault_idxs) -> Dict:
    if not ranks:
        return {}
    arr = np.array(ranks, dtype=float)
    per_ft: Dict[str, List[int]] = {}
    for r, fi in zip(ranks, fault_idxs):
        if 0 <= fi < len(FAULT_TYPES):
            per_ft.setdefault(FAULT_TYPES[fi], []).append(r)
    result = {
        "hit1": float(np.mean(arr == 1)),
        "hit3": float(np.mean(arr <= 3)),
        "hit5": float(np.mean(arr <= 5)),
        "mrr":  float(np.mean(1.0 / arr)),
        "n":    len(ranks),
    }
    if per_ft:
        ft_h1 = {ft: float(np.mean(np.array(v) == 1)) for ft, v in per_ft.items()}
        result["per_fault_hit1"] = ft_h1
        result["macro_hit1"] = float(np.mean(list(ft_h1.values())))
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def build_train(train_idx, feature_fn):
    Xs, ys = [], []
    for idx in train_idx:
        g = load_graph(idx)
        X, y, _ = feature_fn(g)
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    from xgboost import XGBClassifier

    train_idx, val_idx = build_train_val_indices(DATASET_SIZE, 0.80, SEED)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    print("Загрузка val графов...")
    val_graphs = [load_graph(i) for i in val_idx]
    n_faulted = sum(1 for g in val_graphs if
                    hasattr(g, "fault_type_idx") and int(g.fault_type_idx) >= 0
                    and any(g[nt].y.sum() > 0 for nt in RC_CANDIDATE_TYPES
                            if nt in g.node_types))

    results = {}

    # 1. XGBoost — только value канал (нет temporal engineering)
    print("\n[1/4] XGBoost — только VALUE (нет delta_long)...")
    X, y = build_train(train_idx, get_value_only)
    scale = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    xgb1 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=scale, eval_metric="logloss",
                          tree_method="hist", n_jobs=-1, verbosity=0, random_state=42)
    t0 = time.perf_counter()
    xgb1.fit(X, y)
    print(f"  fit: {time.perf_counter()-t0:.1f}s  X.shape={X.shape}")
    m = eval_sklearn(xgb1, val_graphs, get_value_only)
    results["XGBoost (value only, без temporal)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    # 2. XGBoost — полный temporal (delta_long etc)
    print("\n[2/4] XGBoost — полный temporal (с delta_long)...")
    X, y = build_train(train_idx, get_full_temporal)
    scale = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    xgb2 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=scale, eval_metric="logloss",
                          tree_method="hist", n_jobs=-1, verbosity=0, random_state=42)
    t0 = time.perf_counter()
    xgb2.fit(X, y)
    print(f"  fit: {time.perf_counter()-t0:.1f}s  X.shape={X.shape}")
    m = eval_sklearn(xgb2, val_graphs, get_full_temporal)
    results["XGBoost (temporal, без графа)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    # 3. XGBoost — temporal + ручные признаки соседей из графа
    print("\n[3/4] XGBoost — temporal + СОСЕДИ из графа (ручная агрегация)...")
    X, y = build_train(train_idx, get_graph_features)
    scale = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    xgb3 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=scale, eval_metric="logloss",
                          tree_method="hist", n_jobs=-1, verbosity=0, random_state=42)
    t0 = time.perf_counter()
    xgb3.fit(X, y)
    print(f"  fit: {time.perf_counter()-t0:.1f}s  X.shape={X.shape}")
    m = eval_sklearn(xgb3, val_graphs, get_graph_features)
    results["XGBoost (temporal + соседи, ручной граф)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    # 4. GNN + structural probes
    print("\n[4/4] GATv2Hetero (message passing, end-to-end)...")
    m = eval_gnn(val_graphs, mode="normal")
    results["GATv2Hetero (наш GNN, end-to-end)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    print("\n[probe] GNN — NO EDGES (message passing removed at inference)...")
    m = eval_gnn(val_graphs, mode="no_edges")
    results["GNN probe (no edges)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    print("\n[probe] GNN — RANDOM EDGES (topology destroyed, edge counts preserved)...")
    m = eval_gnn(val_graphs, mode="random_edges")
    results["GNN probe (random edges)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    print("\n[probe] GNN — LOCAL ONLY (input projection + SharedScorer, no GAT layers)...")
    m = eval_gnn(val_graphs, mode="local_only")
    results["GNN probe (local only scorer)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}")

    # итог
    print("\n" + "=" * 72)
    print(f"ABLATION STUDY (val n_faulted={n_faulted})")
    print("=" * 72)
    print(f"{'Модель':<45} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6}")
    print("-" * 72)
    for name, m in results.items():
        print(f"  {name:<43} {m['hit1']:>5.1%} {m['hit3']:>5.1%} {m['mrr']:>6.3f}")
    print("=" * 72)

    # per-fault для network_congestion (самый структурный тип)
    print("\nHit@1 для network_congestion (структурный тип — switch без своей аномалии):")
    for name, m in results.items():
        v = m.get("per_fault_hit1", {}).get("network_congestion", 0)
        print(f"  {name:<45} {v:.1%}")

    # запись
    out = ROOT / "reports" / "ablation_study.md"
    _write(results, n_faulted, out)
    print(f"\nОтчёт -> {out}")


def _write(results, n_faulted, out):
    net_row = ""
    lines = [
        "# Ablation study: что реально даёт GNN",
        "",
        f"**val_faulted={n_faulted}  seed=42  dataset=v5a_40**",
        "",
        "## Результаты",
        "",
        "| Модель | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|--------|------:|------:|------:|----:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | **{m['hit1']:.1%}** | {m['hit3']:.1%} "
            f"| {m.get('hit5',0):.1%} | {m['mrr']:.3f} |"
        )
    lines += [
        "", "## network_congestion Hit@1 (самый структурный тип)",
        "", "| Модель | Hit@1 |", "|--------|------:|",
    ]
    for name, m in results.items():
        v = m.get("per_fault_hit1", {}).get("network_congestion", 0)
        lines.append(f"| {name} | {v:.1%} |")

    if "GATv2Hetero (наш GNN, end-to-end)" in results:
        base = results["GATv2Hetero (наш GNN, end-to-end)"]["hit1"]
        lines += [
            "",
            "## Edge-dependence probes",
            "",
            "Эти проверки не переобучают модель: они ломают рёбра только на инференсе.",
            "Если падение маленькое, значит текущий `v5a_40` слишком хорошо решается локальными признаками.",
            "",
            "| Probe | Hit@1 | Delta vs normal GNN |",
            "|-------|------:|--------------------:|",
        ]
        for name in [
            "GNN probe (no edges)",
            "GNN probe (random edges)",
            "GNN probe (local only scorer)",
        ]:
            if name in results:
                h = results[name]["hit1"]
                lines.append(f"| {name} | {h:.1%} | {h - base:+.1%} |")

    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
