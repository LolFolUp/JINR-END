# Интеллектуальная система RCA для HPC-кластера на базе GNN + RAG/LLM

Проект реализует end-to-end pipeline для анализа инцидентов в суперкомпьютерном
кластере: графовая нейронная сеть локализует первопричину сбоя, а RAG/LLM-модуль
строит безопасный план действий для инженера.

Ключевая идея работы: инфраструктура кластера естественно описывается как
гетерогенный граф. Поэтому для каскадных сбоев важно анализировать не только
локальные метрики отдельного узла, но и связи между CPU, GPU, RAM, HDD,
коммутаторами и задачами планировщика.

## Текущий статус

| Направление | Статус |
|---|---|
| GNN RCA model | Реализована inference-модель `GATv2Hetero` |
| Dataset pipeline | Синтетический датасет `v5a_40` и topology stress-test `v6_topology_screen` |
| GNN + LLM интеграция | Реализован адаптер `gnn_to_incident` и demo CLI |
| RAG/LLM remediation | Qdrant/Redis/Ollama + fallback-режимы |
| Safety layer | Semantic Firewall + Action DSL |
| Тесты | `python -m pytest tests/ -v` |
| Документация | Основные отчеты в `reports/` |

## Архитектура системы

```text
HPC telemetry + cluster topology
        |
        v
Graph construction
        |
        v
GNN RCA layer
GATv2Hetero ranks 406 root-cause candidates
        |
        v
GNN -> Incident adapter
converts graph output into incident context
        |
        v
RAG layer
retrieves SOPs, incident history and technical context
        |
        v
LLM remediation layer
generates a structured remediation playbook
        |
        v
Semantic Firewall + Action DSL
validates that recommended actions are safe
        |
        v
Operator-facing report
```

В системе разделены две задачи:

1. **RCA / локализация первопричины**: выполняется GNN на структурированных
   графовых данных.
2. **План устранения**: выполняется RAG/LLM на основе результата GNN,
   регламентов, истории инцидентов и контекста узла.

LLM не заменяет GNN: она не ранжирует 406 кандидатов по телеметрии, а получает
уже найденную GNN первопричину и формирует безопасный playbook.

## Графовая постановка задачи

Каждый снимок состояния кластера представлен как гетерогенный граф:

| Элемент | Описание |
|---|---|
| Node types | `cpu`, `gpu`, `ram`, `hdd`, `switch`, `job`, `rca_context` |
| RC-кандидаты | `cpu`, `gpu`, `ram`, `hdd`, `switch` |
| Число RC-кандидатов | 406 на граф |
| Edge types | 16 типов физических, логических и контекстных связей |
| Feature channels | `value`, `delta_short`, `delta_long`, `rolling_var` |
| Цель | Ранжировать все RC-кандидаты и поставить истинный root cause как можно выше |

Основные метрики:

- **Hit@1 / RCA Top-1**: true root cause стоит на первом месте.
- **Hit@3 / Hit@5**: true root cause попал в топ-3 или топ-5.
- **MRR**: среднее обратное место истинной первопричины.

## GNN-модель

Актуальная inference-модель находится в `gnn/model.py`.

| Параметр | Значение |
|---|---|
| Архитектура | `GATv2Hetero` |
| Слои | 2 heterogeneous GATv2 слоя |
| Hidden dim | 64 |
| Attention heads | 4 |
| Scorer | Shared scorer с embedding типа узла |
| Loss в экспериментах | Global cross-entropy по всем RC-кандидатам графа |
| Checkpoint | `gnn/checkpoints/best_v5a_40_screening.pt` |

Checkpoint-файл не хранится в git как обычный исходный код. Для real inference
положите его в `gnn/checkpoints/` или укажите путь через переменную окружения
`GNN_CHECKPOINT`.

Причина выбора GNN: при каскадных сбоях истинная причина не всегда является
самым аномальным локальным узлом. Она может определяться согласованным паттерном
деградации соседних компонентов. Message passing позволяет передавать сигнал по
физическим и логическим связям кластера.

## Датасеты и эксперименты

В проекте используются два разных экспериментальных режима. Их важно не смешивать.

### `v5a_40`: основной synthetic benchmark

`v5a_40` используется как основной датасет качества RCA pipeline.

| Характеристика | Значение |
|---|---|
| Размер | 4500 графов |
| Healthy ratio | 0.40 |
| Fault types | 9 |
| Validation faulted graphs | 534 |
| Назначение | Проверка общего качества RCA на физически осмысленных synthetic-сценариях |

9 типов неисправностей:

- `hdd_degradation`
- `network_congestion`
- `ram_leak`
- `cpu_frequency_drop`
- `cpu_cache_thrashing`
- `memory_bandwidth_saturation`
- `swap_thrashing`
- `gpu_thermal_throttle`
- `disk_full`

Результаты на `v5a_40`:

| Модель | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| XGBoost value-only | 43.1% | 54.3% | 58.6% | 0.509 |
| XGBoost temporal без топологии | 85.6% | 92.1% | 93.3% | 0.892 |
| XGBoost temporal + manual neighbors | 86.3% | 91.8% | 93.3% | 0.895 |
| **GATv2Hetero** | **87.5%** | **92.3%** | **93.8%** | **0.903** |

Вывод: на локально-разделимых сценариях XGBoost почти догоняет GNN, потому что
получает сильные временные признаки (`delta_long`, `rolling_var`). Это не
опровергает GNN, а показывает качество feature engineering и SNR-аудита датасета.

### Edge-dependence probes на `v5a_40`

Чтобы проверить, использует ли GNN граф, были проведены inference-probes:

| Probe | Hit@1 | Delta vs normal GNN |
|---|---:|---:|
| GNN normal | 87.5% | 0.0 pp |
| GNN no edges | 40.3% | -47.2 pp |
| GNN random edges | 23.8% | -63.7 pp |
| GNN local-only scorer | 54.3% | -33.1 pp |

Вывод: даже на `v5a_40` GNN не является просто локальным MLP. При удалении или
разрушении рёбер качество резко падает.

### `v6_topology_screen`: topology-dependent benchmark

`v6_topology_screen` не заменяет `v5a_40`. Это дополнительный stress-test,
созданный для проверки графовой гипотезы.

Идея benchmark:

- локальный сигнал root cause ослаблен;
- добавлены same-type decoy-узлы;
- решающая информация перенесена в согласованный паттерн топологически связанных
  victim-узлов;
- структура графа сохранена.

Результаты на расширенном прогоне `700 train / 300 val`:

| Модель | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| XGBoost value-only | 28.0% | 69.3% | 100.0% | 0.533 |
| XGBoost temporal без топологии | 28.0% | 71.0% | 100.0% | 0.535 |
| MLP local-only | 25.3% | 73.0% | 100.0% | 0.516 |
| **GNN full graph** | **89.7%** | **99.7%** | **100.0%** | **0.946** |
| GNN no-edge probe | 0.0% | 0.3% | 6.7% | 0.048 |
| GNN random-edge probe | 2.0% | 4.7% | 6.3% | 0.048 |
| XGBoost temporal + manual neighbors | 94.3% | 99.7% | 100.0% | 0.970 |

Главный вывод:

> В topology-dependent сценариях XGBoost без знания связей кластера определяет
> первопричину только в 28.0% случаев, тогда как GNN, использующая граф
> топологии, достигает 89.7% RCA Top-1.

`XGBoost temporal + manual neighbors` является сильным baseline, но это уже не
обычная табличная модель. Ему вручную передаются mean/max признаки соседей из
`edge_index`, то есть топология заранее кодируется инженером. В дипломной
аргументации его корректно трактовать как upper-bound для ручной topology
feature engineering.

## Как читать сравнение с XGBoost

В работе важно различать три постановки:

| Постановка | Что видит модель | Интерпретация |
|---|---|---|
| XGBoost temporal | Только признаки отдельного узла | Честный табличный baseline без топологии |
| XGBoost manual neighbors | Признаки узла + вручную агрегированные признаки соседей | Сильный baseline с ручной топологической инженерией |
| GNN | Признаки узлов + `edge_index` + `edge_attr` | Автоматическое обучение агрегации по графу |

Тезис для диплома:

> На локальных сбоях табличные методы могут быть близки к GNN, если признаки
> хорошо спроектированы. На каскадных topology-dependent сбоях табличная модель
> без топологии теряет информацию, а GNN сохраняет высокое качество за счёт
> message passing.

## RAG/LLM-модуль

После GNN inference результат преобразуется в incident contract:

```text
GNN output
  root_cause node
  top-k candidates
  key anomalous metrics
  affected topology neighborhood
        |
        v
Incident context for RAG/LLM
```

Компоненты:

| Компонент | Назначение |
|---|---|
| `gnn/inference.py` | Стандартизированный inference API для GNN |
| `integrations/gnn_to_incident.py` | Адаптер GNN-output в incident contract |
| `rag/` | Поиск SOP и контекста |
| `llm/prompt_builder.py` | Prompt с guardrails: LLM не должна менять root cause без оснований |
| `remediation/firewall.py` | Semantic Firewall и проверка Action DSL |
| `app/demo_gnn_llm_pipeline.py` | End-to-end demo CLI |

LLM-слой не получает права свободно выполнять команды. Он генерирует
структурированный playbook, который проходит валидацию через firewall и Action
DSL.

## Быстрый запуск

### Offline demo без Docker

```bash
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt --mock
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_11.pt --mock
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_2.pt --mock
```

### Только GNN inference

```bash
python -m gnn.inference --sample demo_data/gnn_samples/data_3.pt --top-k 5
```

### Real mode с RAG/LLM сервисами

Нужны поднятые Ollama, Qdrant и Redis:

```bash
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt
```

### Тесты

```bash
python -m pytest tests/ -v
```

### Пересчёт benchmark-отчётов

```bash
python scripts/ablation_study.py
python scripts/structural_benchmark.py --max-train-graphs 700 --max-val-graphs 300 --epochs 12 --mlp-epochs 3
```

## Структура репозитория

```text
JINR-rag/
├── app/
│   └── demo_gnn_llm_pipeline.py       # end-to-end demo GNN -> RAG/LLM
├── gnn/
│   ├── model.py                       # inference-time GATv2Hetero
│   ├── inference.py                   # standardized GNN inference API
│   └── checkpoints/                   # trained checkpoint
├── integrations/
│   └── gnn_to_incident.py             # adapter from GNN output to incident context
├── training_pipeline/
│   ├── config.py                      # topology and feature schema
│   ├── dataset_generator.py           # synthetic dataset generation line
│   ├── eval_utils.py                  # shared split and RCA metrics
│   └── train.py                       # training pipeline from experiment line
├── rag/                               # retrieval layer
├── llm/                               # LLM client and prompt builder
├── remediation/                       # firewall, Action DSL, remediation pipeline
├── edge-agent/                        # Go telemetry agent (L1–L2)
├── snapshot_engine/                   # Kafka → snapshot → GNN (L3–L4 hook)
├── e2e_simulator/                     # end-to-end smoke tests
├── proto/                             # telemetry protobuf bindings
├── scripts/
│   ├── ablation_study.py              # v5a_40 baselines and graph probes
│   └── structural_benchmark.py        # v6_topology_screen stress-test
├── dataset/
│   └── v6_topology_screen/            # metadata for generated topology stress-test
├── demo_data/
│   └── gnn_samples/                   # small demo graphs
├── reports/
│   ├── ablation_study.md
│   ├── baseline_comparison.md
│   ├── structural_benchmark.md
│   ├── gnn_llm_end_to_end_integration_ru.md
│   └── gnn_dataset_pipeline_full_readme_ru.md
└── tests/
    ├── test_gnn_integration.py
    └── test_rag_pipeline.py
```

## Документация

| Файл | Назначение |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Software architecture, графовая модель, домен HPC |
| [`LAYERS.md`](LAYERS.md) | Послойная модель pipeline (GNN → RAG → LLM → firewall) |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Локальный запуск, Docker Compose, troubleshooting |
| [`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md) | Куда вынесены legacy-компоненты |

## Telemetry pipeline (L1–L3)

Live-контур сбора и сборки graph snapshot **в репозитории**:

| Путь | Роль |
|---|---|
| `edge-agent/` | Go Edge-agent: сбор телеметрии, L2 feature processing, Kafka/Protobuf |
| `snapshot_engine/` | Kafka consumer, Polars join, HeteroData snapshot, GNN inference hook |
| `e2e_simulator/` | mock producer + smoke tests |
| `proto/` | Protobuf bindings для Python |

Kafka-сервис в `docker-compose.yml` используется этим контуром.

## Legacy archive

Устаревшие root-доки, phase-артефакты и v3 comparison-скрипты вынесены на диск:
`D:\Vlad\JINR-rag-archive\legacy-2026-05-31\` — см. [`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md).

## Основные отчёты

| Файл | Назначение |
|---|---|
| `reports/gnn_dataset_pipeline_full_readme_ru.md` | Полное описание генерации данных и GNN pipeline |
| `reports/ablation_study.md` | Сравнение GNN с XGBoost и edge-dependence probes |
| `reports/baseline_comparison.md` | Сводное сравнение с табличными baseline |
| `reports/structural_benchmark.md` | Topology-dependent benchmark |
| `reports/gnn_llm_end_to_end_integration_ru.md` | Документация GNN + RAG/LLM integration |

## Ограничения и честность результатов

Результаты на синтетике нельзя напрямую трактовать как production-качество на
реальном суперкомпьютере.

Ограничения текущей версии:

- `v5a_40` и `v6_topology_screen` являются synthetic benchmarks.
- `fault_type_hint` в demo берётся из synthetic ground truth; в production нужен
  отдельный fault classifier.
- `node_id` является синтетическим индексом; для реального стенда нужен CMDB или
  inventory lookup.
- `score = sigmoid(logit)` не является калиброванной вероятностью.
- Реальная телеметрия Говоруна может содержать пропуски, смешанные инциденты,
  шумные labels и отличающиеся режимы нагрузки.

Корректная формулировка статуса:

> Система готова к пилотной проверке на тестовом стенде: необходимо собрать
> реальные временные ряды, топологию, журналы инцидентов и экспертную разметку
> root cause, после чего проверить переносимость закономерностей синтетического
> датасета на данные суперкомпьютера Говорун.

## Короткая формулировка для диплома

Разработана интеллектуальная система анализа первопричин инцидентов в
суперкомпьютерном кластере, объединяющая гетерогенную графовую нейронную сеть и
RAG/LLM-модуль генерации рекомендаций. GNN решает задачу ранжирования
root-cause кандидатов на графе инфраструктуры, а LLM формирует безопасный план
устранения на основе SOP, истории инцидентов и технического контекста. На
основном synthetic benchmark `v5a_40` модель достигает 87.5% Hit@1, а на
topology-dependent benchmark показывает 89.7% Hit@1 против 28.0% у XGBoost без
топологии, что подтверждает ценность message passing для каскадных сбоев.

## Репродуцируемые команды

```bash
# Full test suite
python -m pytest tests/ -v

# GNN-only inference
python -m gnn.inference --sample demo_data/gnn_samples/data_3.pt --top-k 5

# End-to-end mock demo
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt --mock

# v5a_40 ablation
python scripts/ablation_study.py

# topology-dependent benchmark
python scripts/structural_benchmark.py --max-train-graphs 700 --max-val-graphs 300 --epochs 12 --mlp-epochs 3
```

`dataset/v6_topology_screen/raw/*.pt` не хранится в git: это воспроизводимый
generated artifact. Скрипт выше пересоздаёт raw-графы и обновляет
`reports/structural_benchmark.md`.

## Citation

```bibtex
@thesis{sivolapov2026_gnn_rca,
  author = {Sivolapov, Vladislav},
  title = {Root Cause Analysis in HPC Clusters using Graph Neural Networks and Retrieval-Augmented Generation},
  school = {MISIS University},
  year = {2026}
}
```
