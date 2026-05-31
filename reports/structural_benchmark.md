# Structural benchmark: доказательство графового преимущества GNN

**base_dataset:** `d:\Vlad\JINR-rag-exp-rca70\dataset\v5a_40\raw`
**structural_dataset:** `D:\Vlad\JINR-rag\dataset\v6_topology_screen`
**train_graphs:** 700  **val_graphs:** 300
**screening_epochs:** 12  **seed:** 42

## Результаты

| Модель | Hit@1 | Hit@3 | Hit@5 | MRR |
|--------|------:|------:|------:|----:|
| XGBoost value-only | **28.0%** | 69.3% | 100.0% | 0.533 |
| XGBoost temporal | **28.0%** | 71.0% | 100.0% | 0.535 |
| XGBoost temporal + manual neighbors | **94.3%** | 99.7% | 100.0% | 0.970 |
| MLP local-only | **25.3%** | 73.0% | 100.0% | 0.516 |
| GNN full graph | **89.7%** | 99.7% | 100.0% | 0.946 |
| GNN no-edge probe | **0.0%** | 0.3% | 6.7% | 0.048 |
| GNN random-edge probe | **2.0%** | 4.7% | 6.3% | 0.048 |

## Главные дельты

- `GNN full graph - XGBoost temporal`: **+61.7% Hit@1**.
- `GNN full graph - GNN no-edge`: **+89.7% Hit@1**.
- `GNN full graph - GNN random-edge`: **+87.7% Hit@1**.
- `GNN full graph - MLP local-only`: **+64.3% Hit@1**.

## Интерпретация

`v5a_40` остаётся основным high-quality synthetic benchmark для качества RCA pipeline.
Этот structural benchmark отдельно проверяет графовую гипотезу: RC локально ослаблен, рядом есть decoy того же типа, а решающая информация распределена по топологически связанным жертвам.
Поэтому падение `no-edge/random-edge` относительно full graph является прямым доказательством, что модель использует связи, а не только локальные temporal-признаки.

Ручной `XGBoost temporal + manual neighbors` получает 94.3% Hit@1 и в этом stress-test выше GNN. Это не опровергает графовую гипотезу: этот baseline получает специально сконструированные mean/max признаки соседей из `edge_index`, то есть вручную закодированную топологию. В дипломе его нужно трактовать как инженерно дорогой upper-bound для табличных методов, а не как обычный локальный baseline.

Главный доказательный результат: без графа локальные методы дают 25.3-28.0% Hit@1, тогда как GNN с message passing даёт 89.7% Hit@1, а та же GNN при удалении или рандомизации рёбер почти полностью деградирует.