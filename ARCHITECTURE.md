# Архитектура системы JINR-rag

Документ описывает **текущую** архитектуру репозитория: telemetry pipeline L1–L3,
GNN RCA (L4), RAG/LLM remediation (L5–L6). Phase-артефакты и старые root-доки —
в [`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md).

Подробные метрики экспериментов — в [`README.md`](README.md) и [`reports/`](reports/).
Схема признаков и топологии — в [`training_pipeline/config.py`](training_pipeline/config.py).

---

## 1. Назначение системы

Система решает две разделённые задачи:

| Задача | Слой | Вход | Выход |
|---|---|---|---|
| **RCA / локализация первопричины** | GNN | Гетерогенный граф телеметрии | Ранжирование 406 RC-кандидатов |
| **План устранения** | RAG + LLM + Firewall | Incident contract от GNN | Безопасный remediation playbook |

LLM **не** заменяет GNN: она не ранжирует кандидатов по телеметрии, а получает
уже найденную первопричину и формирует действия по SOP и контексту.

---

## 2. Архитектура ПО (текущий стек)

```text
  edge-agent (Go) ──Kafka/Protobuf──► snapshot_engine (Polars)
        |                                      |
        |                                      v
        |                              HeteroData snapshot
        |                                      |
        +──────── demo_data/*.pt ────────────┘
                              |
                              v
                    +---------------------+
                    |  gnn/inference.py   |
                    |  GATv2Hetero        |
                    |  Hit@1 on v5a_40    |
                    +---------------------+
                              |
                              v
                    +---------------------------+
                    | integrations/             |
                    | gnn_to_incident.py        |
                    | inference JSON contract   |
                    +---------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    app/demo_gnn_llm_pipeline.py      remediation/run.py
    (полный E2E demo)                 (только RAG/LLM,
              |                        из inference_sample.json)
              v
    +---------+---------+---------+---------+
    |   RAG   |   LLM   | Firewall| Report  |
    | rag/    | llm/    | remed.  | remed.  |
    +---------+---------+---------+---------+
              |
              v
    artifacts/remediation_report.json
    artifacts/gnn_llm_demo_trace.json
```

### Ключевые модули

| Путь | Роль |
|---|---|
| [`gnn/model.py`](gnn/model.py) | Inference-time `GATv2Hetero` (2 слоя, hidden=64, 4 heads) |
| [`gnn/inference.py`](gnn/inference.py) | Стандартизированный API inference, top-k, метрики |
| [`integrations/gnn_to_incident.py`](integrations/gnn_to_incident.py) | GNN output → incident / inference contract |
| [`app/demo_gnn_llm_pipeline.py`](app/demo_gnn_llm_pipeline.py) | End-to-end CLI: graph → GNN → RAG → LLM → firewall |
| [`remediation/pipeline.py`](remediation/pipeline.py) | Оркестрация RAG/LLM слоёв |
| [`remediation/firewall.py`](remediation/firewall.py) | Semantic Firewall + Action DSL |
| [`rag/`](rag/) | Qdrant, Redis context, SOP knowledge base |
| [`llm/`](llm/) | Ollama client, prompt builder, response parser |
| [`training_pipeline/`](training_pipeline/) | Генерация synthetic dataset, обучение, eval |
| [`scripts/ablation_study.py`](scripts/ablation_study.py) | Benchmark `v5a_40`, XGBoost baselines, edge probes |
| [`scripts/structural_benchmark.py`](scripts/structural_benchmark.py) | Topology stress-test `v6_topology_screen` |
| [`edge-agent/`](edge-agent/) | Go telemetry agent, L1–L2 feature processing |
| [`snapshot_engine/`](snapshot_engine/) | Kafka consumer, snapshot assembly, inference hook |
| [`e2e_simulator/`](e2e_simulator/) | Mock producer и smoke tests |
| [`proto/`](proto/) | Protobuf schema/bindings |
| [`api/grafana_api.py`](api/grafana_api.py) | REST API для Grafana dashboards (опционально) |

### Checkpoint и артефакты

| Артефакт | Путь | В git |
|---|---|---|
| GNN checkpoint | `gnn/checkpoints/best_v5a_40_screening.pt` | нет (положить локально или через `GNN_CHECKPOINT`) |
| Demo graphs | `demo_data/gnn_samples/*.pt` | да |
| Inference sample | `artifacts/inference_sample.json` | да |
| Demo trace | `artifacts/gnn_llm_demo_trace.json` | да |
| v6 raw graphs | `dataset/v6_topology_screen/raw/*.pt` | нет (генерируются скриптом) |

---

## 3. Графовая постановка RCA

Каждый снимок состояния кластера — **гетерогенный граф** PyTorch Geometric `HeteroData`.

| Элемент | Значение |
|---|---|
| Node types | `cpu`, `gpu`, `ram`, `hdd`, `switch`, `job`, `rca_context` |
| RC-кандидаты | `cpu`, `gpu`, `ram`, `hdd`, `switch` (406 узлов на граф) |
| Edge types | 16 типов физических, логических и контекстных связей |
| Feature channels | `value`, `delta_short`, `delta_long`, `rolling_var` |
| Цель | Ранжировать RC-кандидатов; true root cause — как можно выше |

### Типы рёбер (схема)

Физическая топология Spine-Leaf и связи железа:

```text
leaf_switch  <--uplink/downlink-->  spine_switch
cpu/gpu/ram/hdd  <--network_via/routes_to-->  leaf_switch
cpu  <--shares_board_with-->  gpu
cpu  <--addresses-->  ram
cpu  <--manages_io-->  hdd
job  <--executes_on/allocates/uses_gpu-->  cpu/ram/gpu
```

Полный список edge types и размерности признаков — в
[`training_pipeline/config.py`](training_pipeline/config.py).

### Временная динамика признаков

Для непрерывных метрик каждый канал представлен не одним числом, а набором:

- **value** — текущее значение телеметрии;
- **delta_short** — импульс за такт (≈5 с);
- **delta_long** — отклонение от EMA-фона (окно ~5 мин);
- **rolling_var** — дисперсия в скользящем окне.

Категориальные признаки (`*_encoded`, `*_flag`, `*_status`) не проходят через EMA.
Нормализация — по global healthy baseline, не по одному аварийному графу.

Генератор реализован в [`training_pipeline/dataset_generator.py`](training_pipeline/dataset_generator.py).

---

## 4. Доменная модель HPC «Говорун»

Ниже — **предметная область**, которую моделирует synthetic dataset. Это не описание
production deployment на реальном кластере.

### 4.1. Сетевая топология

- Архитектура: **Spine-Leaf (Fat-Tree)**, не 3D-Torus.
- Spine: корневые коммутаторы с failover.
- Leaf: подключение серверов и СХД.
- Интерконнект: InfiniBand 100 Gbps, Intel Omni-Path, Ethernet для управления и GPU-узлов.

### 4.2. Типы аппаратных узлов

| Тип | Характеристика |
|---|---|
| Cascade Lake / H04 | 192 GB RAM, Intel Optane PMEM |
| Intel Gold + СХД | 512 GB RAM, IB 100 Gbps, 4×2 TB |
| AMD EPYC + A100 | 5 серверов, 8× A100, Ethernet |
| NVIDIA H100 Hopper | 2 сервера, 8× H100, NVLink |

### 4.3. Логическое разделение кластера

| Зона | Модель в графе |
|---|---|
| Bare-Metal (научные расчёты) | `job → hosted_on → compute_node` (без VM) |
| Service VMs (мониторинг, БД) | `service → vm → management_node` |

### 4.4. Классы сущностей и метрик

В графе моделируются сущности с телеметрией:

| Сущность | Примеры метрик |
|---|---|
| **CPU** | `cpu_usage_total_percent`, `cpu_iowait_time`, `cpu_throttling_events`, IPC |
| **GPU** | `gpu_core_utilization_percent`, `gpu_temperature_celsius`, XID errors, NVLink |
| **RAM / PMEM** | `ram_*`, `pmem_*`, OOM, NUMA hit/miss |
| **HDD / NVMe** | IOPS, latency, SMART, Lustre MDT/OST |
| **Network link** | utilization, drops, CRC, optical power |
| **Switch** | fabric utilization, buffer exhaustion, packet drops |
| **Job (SLURM)** | status, CPU/RAM/I/O/MPI профиль, barrier wait |

Полный перечень имён признаков и правила парсинга continuous vs categorical —
в [`training_pipeline/config.py`](training_pipeline/config.py) и
[`reports/gnn_dataset_pipeline_full_readme_ru.md`](reports/gnn_dataset_pipeline_full_readme_ru.md).

### 4.5. Типовые synthetic-сценарии сбоев

| Сценарий | Механизм | Задача GNN |
|---|---|---|
| Отказ Spine / congestion | packet_drop → I/O wait → стагнация jobs | Найти коммутатор, не «виноватый» сервер |
| PMEM anomaly | падение RAM → OOM на job | Связать job с аппаратной метрикой PMEM |
| GPU thermal throttle | рост temp → throttling → 3× runtime | Показать аппаратную, а не software причину |
| HDD degradation | latency ↑ → job I/O wait → MPI stragglers | Каскад через `manages_io` и MPI edges |

9 типов fault injection в benchmark `v5a_40`: `hdd_degradation`, `network_congestion`,
`ram_leak`, `cpu_frequency_drop`, `cpu_cache_thrashing`, `memory_bandwidth_saturation`,
`swap_thrashing`, `gpu_thermal_throttle`, `disk_full`.

---

## 5. Экспериментальные режимы

Важно не смешивать два benchmark:

| Dataset | Назначение | GNN Hit@1 | XGBoost без топологии |
|---|---|---:|---:|
| `v5a_40` | Общее качество RCA + feature engineering | 87.5% | 85.6% |
| `v6_topology_screen` | Topology-dependent stress-test | 89.7% | 28.0% |

Edge probes на `v5a_40` (no edges → 40.3%, random edges → 23.8%) и `v6` (→ ~0%)
подтверждают зависимость GNN от структуры графа.

Отчёты: [`reports/ablation_study.md`](reports/ablation_study.md),
[`reports/structural_benchmark.md`](reports/structural_benchmark.md).

---

## 6. RAG/LLM слой (incident contract)

После GNN inference адаптер формирует structured context:

```text
root_cause node + confidence
top-k candidates
key anomalous metrics
affected topology neighborhood
fault_type (в demo — из synthetic ground truth)
        |
        v
RAG: SOP chunks + Redis context + history tickets
        |
        v
LLM: structured playbook (JSON)
        |
        v
Firewall: whitelist actions, block shell injection
        |
        v
Operator report
```

Guardrails: LLM не должна менять root cause без оснований; опасные команды
(`rm -rf`, `kill *`, …) блокируются firewall.

---

## 7. Опциональная observability-обвязка

Через `docker-compose.yml` можно поднять (не обязательно для core pipeline):

| Сервис | Порт | Назначение |
|---|---|---|
| `jinr_api` | 8080 | FastAPI → Grafana Infinity datasource |
| `grafana` | 3000 | Node Graph dashboard, feedback confirm/reject |
| `kafka` | 9092 | Telemetry bus для `edge-agent` → `snapshot_engine` |

---

## 8. Ограничения

- Benchmarks синтетические; production-качество на Говоруне не доказано.
- `fault_type_hint` в demo — из ground truth; нужен отдельный classifier.
- `node_id` — синтетический индекс; в production нужен CMDB lookup.
- Score GNN — rank logit, не калиброванная вероятность.

---

## 9. Связанные документы

| Документ | Содержание |
|---|---|
| [`README.md`](README.md) | Главный обзор, команды, метрики |
| [`LAYERS.md`](LAYERS.md) | Послойная модель pipeline |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Docker и локальный запуск |
| [`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md) | Архив legacy-компонентов |
| [`reports/`](reports/) | Экспериментальные отчёты |
