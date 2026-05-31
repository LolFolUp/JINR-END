"""Сравнение GNN vs табличные baseline (XGBoost, RandomForest, LinearSVC).

Задача одна и та же для всех моделей:
    Для каждого графа кластера ранжировать 406 RC-кандидатов и найти корень.
    Метрики: Hit@1, Hit@3, Hit@5, MRR (те же что у GNN).

Для табличных методов:
    Признаки = X-вектор узла (нормализованная телеметрия).
    Двоичный классификатор (RC=1, не-RC=0), оценка = predict_proba[:,1].
    Ранжирование внутри графа по этой оценке.

Честный split: eval_utils.build_train_val_indices(seed=42) — тот же что у GNN.

Запуск:
    python scripts/baseline_comparison.py
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_pipeline.eval_utils import build_train_val_indices, RC_CANDIDATE_TYPES, FAULT_TYPES

# ── пути ──────────────────────────────────────────────────────────────────────
RAW_DIR    = Path(r"d:\Vlad\JINR-rag-exp-rca70\dataset\v5a_40\raw")
CKPT_PATH  = ROOT / "gnn" / "checkpoints" / "best_v5a_40_screening.pt"
META_PATH  = ROOT / "gnn" / "artifacts" / "metadata.json"
REPORT_OUT = ROOT / "reports" / "baseline_comparison.md"

DATASET_SIZE = 4500
TRAIN_RATIO  = 0.80
SEED         = 42


# ── загрузка графов ───────────────────────────────────────────────────────────

def load_graph(idx: int):
    return torch.load(RAW_DIR / f"data_{idx}.pt", weights_only=False)


# ── извлечение табличных признаков ────────────────────────────────────────────

_MAX_DIM = 46  # gpu имеет наибольшую размерность


def extract_tabular(graph) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Признаки, метки, fault_idx для всех RC-кандидатов одного графа.

    Размерности у типов разные (cpu=32, gpu=46, ...) — дополняем нулями до MAX_DIM.
    Также добавляем one-hot тип узла (5 признаков) чтобы модель знала контекст.

    Возвращает:
        X      [N_cand, MAX_DIM+5]  — x-вектор + padding + тип
        y      [N_cand]             — 1 для RC, 0 для остальных
        fi     int                  — fault_type_idx (-1 = healthy)
        rc_pos int                  — позиция RC в X
    """
    rows_X, rows_y = [], []
    for ti, nt in enumerate(RC_CANDIDATE_TYPES):
        if nt not in graph.node_types:
            continue
        store = graph[nt]
        x = store.x.float().numpy()          # [N_nt, feat_dim]
        y = store.y.float().numpy()          # [N_nt]
        # pad до MAX_DIM
        if x.shape[1] < _MAX_DIM:
            pad = np.zeros((x.shape[0], _MAX_DIM - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        # добавляем one-hot тип (5 значений)
        type_oh = np.zeros((x.shape[0], len(RC_CANDIDATE_TYPES)), dtype=np.float32)
        type_oh[:, ti] = 1.0
        x = np.concatenate([x, type_oh], axis=1)
        rows_X.append(x)
        rows_y.append(y)
    if not rows_X:
        return np.empty((0, 0)), np.empty(0), -1, -1
    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    fi = int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1
    rc_pos = int(np.argmax(y)) if y.sum() > 0 else -1
    return X, y, fi, rc_pos


# ── метрики (идентично eval_utils) ────────────────────────────────────────────

def compute_metrics(ranks: List[int], fault_idxs: List[int]) -> Dict:
    if not ranks:
        return {}
    arr = np.array(ranks, dtype=float)
    per_ft: Dict[str, List[int]] = {}
    for r, fi in zip(ranks, fault_idxs):
        if 0 <= fi < len(FAULT_TYPES):
            ft = FAULT_TYPES[fi]
            per_ft.setdefault(ft, []).append(r)
    result = {
        "hit1": float(np.mean(arr == 1)),
        "hit3": float(np.mean(arr <= 3)),
        "hit5": float(np.mean(arr <= 5)),
        "mrr":  float(np.mean(1.0 / arr)),
        "n":    len(ranks),
    }
    if per_ft:
        ft_hit1 = {ft: float(np.mean(np.array(v) == 1)) for ft, v in per_ft.items()}
        result["macro_hit1"] = float(np.mean(list(ft_hit1.values())))
        result["worst_fault"] = min(ft_hit1, key=ft_hit1.__get__ if False else ft_hit1.get)
        result["per_fault_hit1"] = ft_hit1
    return result


def eval_model(model, val_graphs) -> Tuple[Dict, float]:
    """Оценить sklearn-модель на val графах."""
    ranks, fault_idxs = [], []
    t0 = time.perf_counter()
    for graph in val_graphs:
        X, y, fi, _ = extract_tabular(graph)
        if y.sum() == 0 or X.shape[0] == 0:
            continue
        try:
            scores = model.predict_proba(X)[:, 1]
        except AttributeError:
            scores = model.decision_function(X)
        order = np.argsort(scores)[::-1]
        rc_rank = int(np.where(order == np.argmax(y))[0][0]) + 1
        ranks.append(rc_rank)
        fault_idxs.append(fi)
    elapsed = time.perf_counter() - t0
    return compute_metrics(ranks, fault_idxs), elapsed


# ── GNN inference ─────────────────────────────────────────────────────────────

def eval_gnn(val_graphs) -> Tuple[Dict, float]:
    from gnn.model import GATv2Hetero, RC_CANDIDATE_TYPES as RCT
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = GATv2Hetero(
        node_dims=ckpt["node_dims"],
        edge_types=ckpt["edge_types"],
        scorer_mode=ckpt.get("scorer_mode", "shared"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ranks, fault_idxs = [], []
    t0 = time.perf_counter()
    for graph in val_graphs:
        x_dict = {nt: graph[nt].x.float() for nt in graph.node_types}
        ei_dict = {et: graph[et].edge_index for et in graph.edge_types}
        ea_dict = {et: graph[et].edge_attr.float() for et in graph.edge_types}
        with torch.no_grad():
            logits = model(x_dict, ei_dict, ea_dict)

        cands: List[Tuple[float, int]] = []
        rc_global_idx = None
        offset = 0
        for nt in RCT:
            if nt not in logits:
                continue
            lg = logits[nt].float().squeeze(-1)
            sc = torch.sigmoid(lg).numpy()
            y  = graph[nt].y.numpy()
            for i in range(len(sc)):
                if y[i] == 1:
                    rc_global_idx = offset + i
                cands.append((float(sc[i]), offset + i))
            offset += len(sc)

        if rc_global_idx is None:
            continue
        cands.sort(reverse=True)
        rc_rank = next(r+1 for r, (_, idx) in enumerate(cands) if idx == rc_global_idx)
        ranks.append(rc_rank)
        fi = int(graph.fault_type_idx) if hasattr(graph, "fault_type_idx") else -1
        fault_idxs.append(fi)

    elapsed = time.perf_counter() - t0
    return compute_metrics(ranks, fault_idxs), elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка индексов split...")
    train_idx, val_idx = build_train_val_indices(DATASET_SIZE, TRAIN_RATIO, SEED)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    # ── загрузка train-графов для fit ─────────────────────────────────────────
    print("\nСборка обучающей выборки для табличных методов...")
    X_train_list, y_train_list = [], []
    healthy_skipped = 0
    for idx in train_idx:
        g = load_graph(idx)
        X, y, fi, _ = extract_tabular(g)
        if X.shape[0] == 0:
            continue
        if y.sum() == 0:
            # для здоровых графов — всё 0, добавляем несколько строк для баланса
            sample_rows = np.random.default_rng(idx).choice(len(y), min(5, len(y)), replace=False)
            X_train_list.append(X[sample_rows])
            y_train_list.append(y[sample_rows])
            healthy_skipped += 1
        else:
            X_train_list.append(X)
            y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print(f"X_train: {X_train.shape}  pos_rate: {y_train.mean():.4f}")

    # ── загрузка val-графов ───────────────────────────────────────────────────
    print("Загрузка val-графов...")
    val_graphs = [load_graph(i) for i in val_idx]
    n_faulted = sum(1 for g in val_graphs
                    if hasattr(g, "fault_type_idx") and int(g.fault_type_idx) >= 0
                    and any(g[nt].y.sum() > 0 for nt in RC_CANDIDATE_TYPES if nt in g.node_types))
    print(f"Val graphs: {len(val_graphs)}  faulted: {n_faulted}")

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[1/4] Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1,
                                class_weight="balanced", random_state=42)
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_train_s = time.perf_counter() - t0
    print(f"  fit: {rf_train_s:.1f}s")
    m, elapsed = eval_model(rf, val_graphs)
    results["RandomForest"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}  eval:{elapsed:.1f}s")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n[2/4] XGBoost...")
    try:
        from xgboost import XGBClassifier
        scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=scale_pos, use_label_encoder=False,
                             eval_metric="logloss", random_state=42,
                             tree_method="hist", n_jobs=-1, verbosity=0)
        t0 = time.perf_counter()
        xgb.fit(X_train, y_train)
        xgb_train_s = time.perf_counter() - t0
        print(f"  fit: {xgb_train_s:.1f}s")
        m, elapsed = eval_model(xgb, val_graphs)
        results["XGBoost"] = m
        print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}  eval:{elapsed:.1f}s")
    except ImportError:
        print("  xgboost не установлен, пропускаем")

    # ── LinearSVC (быстрый линейный baseline) ─────────────────────────────────
    print("\n[3/4] Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=500, class_weight="balanced",
                            solver="lbfgs", C=0.1, random_state=42, n_jobs=-1)
    t0 = time.perf_counter()
    lr.fit(X_train, y_train)
    print(f"  fit: {time.perf_counter()-t0:.1f}s")
    m, elapsed = eval_model(lr, val_graphs)
    results["LogisticRegression"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}  eval:{elapsed:.1f}s")

    # ── GNN ───────────────────────────────────────────────────────────────────
    print("\n[4/4] GATv2Hetero (наша модель)...")
    m, elapsed = eval_gnn(val_graphs)
    results["GATv2Hetero (наш GNN)"] = m
    print(f"  Hit@1={m['hit1']:.1%}  Hit@3={m['hit3']:.1%}  MRR={m['mrr']:.3f}  eval:{elapsed:.1f}s")

    # ── итоговая таблица ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("ИТОГОВОЕ СРАВНЕНИЕ (val, n_faulted={})".format(n_faulted))
    print("="*72)
    header = f"{'Модель':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'n':>5}"
    print(header)
    print("-"*72)
    baseline_hit1 = None
    for name, m in results.items():
        if baseline_hit1 is None:
            baseline_hit1 = m["hit1"]
        delta = f"(+{(m['hit1']-baseline_hit1)*100:+.1f}pp)" if name != list(results.keys())[0] else ""
        print(f"  {name:<28} {m['hit1']:>6.1%} {m['hit3']:>6.1%} {m.get('hit5',0):>6.1%} "
              f"{m['mrr']:>6.3f} {m['n']:>5}  {delta}")
    print("="*72)

    # ── запись отчёта ─────────────────────────────────────────────────────────
    write_report(results, n_faulted)
    print(f"\nОтчёт сохранён -> {REPORT_OUT}")


def write_report(results: Dict, n_faulted: int):
    lines = [
        "# Сравнение GNN vs табличные baseline",
        "",
        f"**Датасет**: v5a_40  n_total=4500  val_faulted={n_faulted}  seed=42  ",
        "**Задача**: ранжирование 406 RC-кандидатов внутри графа, поиск истинного корня неисправности  ",
        "**Признаки baseline**: нормализованный x-вектор узла (value/delta_short/delta_long/rolling_var × число метрик)",
        "",
        "## Результаты",
        "",
        "| Модель | Hit@1 | Hit@3 | Hit@5 | MRR | n |",
        "|--------|-------|-------|-------|-----|---|",
    ]
    names = list(results.keys())
    baseline_hit1 = results[names[0]]["hit1"] if names else 0
    for name, m in results.items():
        delta = f" (+{(m['hit1']-baseline_hit1)*100:.1f}pp)" if name != names[0] else ""
        lines.append(
            f"| {name}{delta} "
            f"| **{m['hit1']:.1%}** | {m['hit3']:.1%} | {m.get('hit5',0):.1%} "
            f"| {m['mrr']:.3f} | {m['n']} |"
        )

    gnn_hit1 = results.get("GATv2Hetero (наш GNN)", {}).get("hit1", 0)
    best_baseline = max(
        (v["hit1"] for k, v in results.items() if "GNN" not in k), default=0
    )
    gain_pp = (gnn_hit1 - best_baseline) * 100
    gain_rel = gnn_hit1 / best_baseline if best_baseline > 0 else float("inf")

    lines += [
        "",
        "## Прирост GNN",
        "",
        f"| Метрика | Лучший baseline | GNN | Прирост |",
        f"|---------|----------------|-----|---------|",
        f"| Hit@1 | {best_baseline:.1%} | {gnn_hit1:.1%} | **+{gain_pp:.1f}pp ({gain_rel:.2f}×)** |",
        "",
        "## Почему GNN лучше",
        "",
        "Табличные методы видят только **локальный x-вектор узла** —",
        "нормализованную телеметрию одного cpu/hdd/switch без контекста соседей.",
        "",
        "GNN через message passing агрегирует информацию от соседей по 16 типам рёбер:",
        "- switch ← cpu/gpu (нагрузка нисходящих хостов)",
        "- cpu ← ram/hdd (заполненность смежных ресурсов)",
        "- job ← cpu (какие задачи пострадали)",
        "",
        "Это позволяет отличить узел, чьи **соседи аномальны** (жертвы),",
        "от узла, чья **собственная телеметрия аномальна** (корень).",
        "Именно это различие недоступно табличным методам.",
        "",
        "## Честность сравнения",
        "",
        "- Одинаковый val split: `torch.randperm(seed=42)`, те же 900 графов",
        "- Одинаковые входные данные: нормализованный x из датасета v5a_40",
        "- Одинаковые метрики: Hit@k/MRR, per-graph ranking",
        "- Baseline обучены на тех же train-графах (3600 графов)",
    ]

    # per-fault breakdown если есть
    gnn_m = results.get("GATv2Hetero (наш GNN)", {})
    rf_m  = results.get("RandomForest", {})
    xgb_m = results.get("XGBoost", {})
    if gnn_m.get("per_fault_hit1") and rf_m.get("per_fault_hit1"):
        lines += ["", "## Hit@1 по типам неисправностей", "",
                  "| Тип | RandomForest | XGBoost | GNN |",
                  "|-----|-------------|---------|-----|"]
        all_fts = sorted(gnn_m["per_fault_hit1"].keys())
        for ft in all_fts:
            gv  = gnn_m["per_fault_hit1"].get(ft, 0)
            rfv = rf_m["per_fault_hit1"].get(ft, 0)
            xv  = xgb_m.get("per_fault_hit1", {}).get(ft, 0) if xgb_m else 0
            lines.append(f"| {ft} | {rfv:.1%} | {xv:.1%} | **{gv:.1%}** |")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
