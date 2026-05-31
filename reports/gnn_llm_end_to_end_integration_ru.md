# GNN + LLM/RAG: Интеграция end-to-end pipeline

**Ветка**: `integration/gnn-llm-end-to-end`  
**Дата**: 2026-05-31  
**Статус**: реализовано и проверено (23/23 тестов, 3/3 демо-прогонов)

---

## Содержание

1. [Зачем объединяем GNN и LLM](#1-зачем-объединяем-gnn-и-llm)
2. [Что делает GNN](#2-что-делает-gnn)
3. [Что делает RAG/LLM](#3-что-делает-ragllm)
4. [Почему LLM не заменяет GNN](#4-почему-llm-не-заменяет-gnn)
5. [Формат GNN output](#5-формат-gnn-output)
6. [Формат incident context (адаптер)](#6-формат-incident-context-адаптер)
7. [Как работает demo-прогон (поэтапно)](#7-как-работает-demo-прогон-поэтапно)
8. [Как запустить mock mode](#8-как-запустить-mock-mode)
9. [Как запустить real mode](#9-как-запустить-real-mode)
10. [Файлы, добавленные/изменённые в этой ветке](#10-файлы-добавленныеизменённые-в-этой-ветке)
11. [Ограничения текущей реализации](#11-ограничения-текущей-реализации)
12. [Следующие шаги для production](#12-следующие-шаги-для-production)
13. [Приложение: результаты трёх demo-прогонов](#13-приложение-результаты-трёх-demo-прогонов)

---

## 1. Зачем объединяем GNN и LLM

HPC-кластер Говорун (ОИЯИ) производит тысячи телеметрических метрик.  
При инциденте инженер вынужден вручную просматривать логи, метрики, SOPы,
историю тикетов — и принимать решение об устранении под временным давлением.

Архитектура двух слоёв:

| Слой | Инструмент | Задача |
|------|-----------|--------|
| Локализация | GATv2Hetero (GNN) | *Где* сбой: ранжирует 406 RC-кандидатов, выдаёт топ-1 узел + аномальные метрики |
| Устранение | RAG + Mistral (LLM) | *Как* исправить: строит playbook из SOPов, истории тикетов, технического контекста |

Раздельность — принципиальная: GNN работает на структурированных данных графа без текста;
LLM не имеет доступа к сырой телеметрии и не способна самостоятельно ранжировать 406 кандидатов
без галлюцинаций. Каждый слой делает то, для чего обучен.

---

## 2. Что делает GNN

**Модель**: `GATv2Hetero` — 2-слойный гетерогенный Graph Attention Network v2  
**Вход**: граф кластера (907 узлов, 16 типов рёбер), нормализованная телеметрия 4 временными каналами  
**Выход**: per-node логиты → sigmoid-скоры → ранжированный список RC-кандидатов

### Ключевые характеристики (checkpoint `best_v5a_40_screening.pt`)

| Параметр | Значение |
|----------|---------|
| Датасет | v5a_40 (4500 графов, healthy_ratio=0.40) |
| Hit@1 | **87.5%** (+24.2pp vs v4.9) |
| Hit@3 | **92.3%** |
| MRR | **0.903** |
| RC-кандидатов на граф | 406 (cpu×100 + gpu×100 + ram×100 + hdd×100 + switch×6) |
| Инференс на CPU | ~27 ms |
| Параметров | 316,882 |

GNN **локализует** корень (какой конкретный узел), но **не классифицирует** тип неисправности.
`fault_type_hint` в синтетическом датасете берётся из ground-truth метаданных графа;
в production он должен поступать из отдельного fault-classifier.

---

## 3. Что делает RAG/LLM

| Компонент | Реализация |
|-----------|-----------|
| Векторное хранилище SOP | Qdrant in-memory (BOW-fallback если недоступен) |
| Контекст узла | Redis / fakeredis / dict-fallback |
| История тикетов | `HistoryTicketsStore` (in-memory, 9 реальных инцидентов) |
| LLM | Ollama/Mistral → rule-based fallback |
| Firewall | Semantic Firewall + ActionDSL (`pydantic` validation) |

Пайплайн: RAG retrieving SOPs + история → сборка prompt → LLM → firewall validation → RemediationPlaybook.

---

## 4. Почему LLM не заменяет GNN

1. **Масштаб**: LLM не ранжирует 406 node-level кандидатов по структурным признакам.
2. **Галлюцинации**: LLM может придумать "виновника" без телеметрии; GNN детерминирован.
3. **Безопасность**: prompt-инъекции не могут изменить числовые логиты модели.
4. **Latency**: GNN — 27 ms; LLM — секунды/десятки секунд.

Роль LLM в pipeline: **не диагностика, а план устранения** — структурированный playbook
из разрешённых ActionDSL-действий, обоснованный SOPами и историей инцидентов.

В системный промпт добавлены явные guardrails:

```
Do NOT change the root-cause node or fault_type unless an SOP excerpt or the
technical context explicitly contradicts it.
Never emit free-form shell/bash commands, scripts, or destructive operations.
```

---

## 5. Формат GNN output

`gnn.inference.GNNInferenceEngine.run()` возвращает структурированный JSON:

```json
{
  "incident_id": "graph_3_network_congestion",
  "graph_id": 3,
  "source": "gnn",
  "model": {
    "name": "GATv2Hetero",
    "checkpoint": "best_v5a_40_screening.pt",
    "dataset_version": "v5a_40",
    "val_hit1": 0.8745
  },
  "score_semantics": "sigmoid(logit)",
  "rca": {
    "root_cause": {
      "rank": 1,
      "node_type": "switch",
      "node_id": 3,
      "node_label": "S3",
      "score": 1.0,
      "logit": 21.03,
      "fault_type_hint": "network_congestion"
    },
    "top_k": [...],
    "hit_metadata": {
      "candidate_count": 406,
      "rc_candidate_types": ["cpu", "gpu", "ram", "hdd", "switch"]
    }
  },
  "fault_type_hint": {
    "value": "network_congestion",
    "provenance": "synthetic_ground_truth",
    "note": "GNN does not predict fault class. In production derive this from a separate fault classifier."
  },
  "graph_context": {
    "affected_nodes": [...],
    "affected_counts": {"cpu": 22, "gpu": 22, "switch": 2},
    "key_metrics": {
      "switch_packet_loss_percent": 6.04,
      "switch_latency_ms": 5.24,
      "switch_bandwidth_usage_percent": 4.97
    }
  },
  "timing": {"gnn_inference_ms": 27},
  "ground_truth": {"predicted_correct": true}
}
```

**Заметки о честности**:
- `score` = `sigmoid(logit)`, не калиброванная вероятность.
- `affected_nodes` = топологические соседи из edge_index (не ground-truth жертвы).
- `key_metrics` = топ-|delta_long| по RC-узлу (Z-нормализованные отклонения от EMA-базелайна).
- `node_id` — синтетический индекс внутри типа; в production нужен CMDB-lookup.

---

## 6. Формат incident context (адаптер)

`integrations.gnn_to_incident.gnn_to_inference()` конвертирует GNN output
в контракт `run_pipeline()`:

```python
{
  "fault_type": "network_congestion",     # из fault_type_hint
  "rc_node": {"type": "switch", "id": "S3", "host_id": 3},
  "confidence": 1.0,                       # = sigmoid(top logit)
  "rc_logit": 21.03,
  "top5_candidates": [...],
  "victim_nodes": [                        # дедуплицированные хосты
    {"id": "host-002", "type": "host"},
    ...
  ],
  "key_metrics": {"switch_packet_loss_percent": 6.04, ...},
  "affected_counts": {"cpu": 22, "gpu": 22, "switch": 2},
  "gnn_rca": {...},                        # полный incident context
  "gnn_inference_ms": 27
}
```

`gnn_to_inference()` **дедуплицирует** cpu-i и gpu-i (один физический хост)
в один `host-{i:03d}` victim-record, чтобы избежать двойного счёта.

`build_incident_context()` строит человекочитаемый `incident_context` со всеми
полями (run_id, timestamp, top-3, anomalous metrics, affected counts),
который передаётся в prompt builder под ключом `gnn_provenance`.

---

## 7. Как работает demo-прогон (поэтапно)

```
.pt graph
  -> Stage 1: GNNInferenceEngine.run()
      GATv2Hetero forward pass (27 ms)
      ранжирование 406 кандидатов
      извлечение key_metrics (delta_long от RC-узла)
      топологические соседи (из edge_index)
  -> Stage 2: gnn_to_inference() + build_incident_context()
      адаптация в pipeline-контракт
      дедупликация cpu/gpu -> host
  -> Stage 3: run_pipeline()
      IncidentAggregator.aggregate() -> severity / summary
      RedisContextStore.fetch_context() -> hostname, OS, SLA tier
      QdrantStore.retrieve() -> top-3 SOP-chunks
      HistoryTicketsStore.find_similar() -> top-3 исторических тикета
  -> Stage 4: LLMClient.generate()
      Ollama/Mistral -> rule_based_fallback
  -> Stage 5: validate_playbook()
      Semantic Firewall (pydantic ActionDSL)
      BLOCKED -> NOTIFY_OPERATOR fallback
      PASSED -> RemediationPlaybook
  -> Итоговый отчёт + JSON trace
```

---

## 8. Как запустить mock mode

Полностью offline, без внешних сервисов (Qdrant/Redis/Ollama):

```bash
python -m app.demo_gnn_llm_pipeline \
  --sample demo_data/gnn_samples/data_3.pt \
  --mock
```

Или раздельно:

```bash
python -m app.demo_gnn_llm_pipeline \
  --sample demo_data/gnn_samples/data_11.pt \
  --mock-llm \
  --mock-rag
```

Флаги:
- `--mock` = `--mock-llm --mock-rag` (детерминированный rule-based playbook + статические SOP)
- `--mock-llm` = rule-based playbook вместо Ollama
- `--mock-rag` = статический SOP-стаб вместо BOW/Qdrant embedder

---

## 9. Как запустить real mode

При запущенных Docker-сервисах (Qdrant + Redis + Ollama):

```bash
# Запуск инфраструктуры
docker-compose up -d qdrant redis ollama

# Real mode (real Qdrant + real Ollama/Mistral)
python -m app.demo_gnn_llm_pipeline \
  --sample demo_data/gnn_samples/data_3.pt

# Прямой GNN inference без pipeline
python -m gnn.inference \
  --sample demo_data/gnn_samples/data_3.pt \
  --top-k 5 \
  --output artifacts/gnn_out.json
```

Graceful degradation:
- Ollama недоступен → `rule_based_fallback` (LLM backend = "rule_based_fallback")
- Qdrant недоступен → `bow_fallback` (retrieval_method = "bow_fallback")
- Redis недоступен → `fakeredis` или `dict_fallback` (context_source)

Все три деградации происходят автоматически без изменения кода.

---

## 10. Файлы, добавленные/изменённые в этой ветке

### Новые файлы

| Файл | Описание |
|------|---------|
| `gnn/__init__.py` | Пакет GNN inference |
| `gnn/model.py` | Self-contained GATv2Hetero + SharedScorer (без training-loop) |
| `gnn/inference.py` | `GNNInferenceEngine` — load checkpoint, forward pass, top-k ranking, CLI |
| `gnn/artifacts/metadata.json` | Метаданные датасета v5a_40 (feature dims, edge types, …) |
| `gnn/artifacts/loss_config.json` | pos_weight конфигурация |
| `gnn/checkpoints/best_v5a_40_screening.pt` | Checkpoint (Hit@1=87.5%, epoch 7) |
| `integrations/__init__.py` | Пакет адаптеров |
| `integrations/gnn_to_incident.py` | Адаптер GNN output → pipeline inference-контракт |
| `app/__init__.py` | Пакет приложения |
| `app/demo_gnn_llm_pipeline.py` | End-to-end demo CLI (4 стадии, mock/real modes) |
| `demo_data/gnn_samples/data_2.pt` | Demo граф: ram_leak (RAM-22) |
| `demo_data/gnn_samples/data_3.pt` | Demo граф: network_congestion (S3) |
| `demo_data/gnn_samples/data_11.pt` | Demo граф: hdd_degradation (HDD-22) |
| `tests/test_gnn_integration.py` | 9 тестов: адаптер, prompt, firewall, e2e mock, real inference |
| `reports/gnn_llm_end_to_end_integration_ru.md` | Этот документ |

### Изменённые файлы

| Файл | Изменение |
|------|----------|
| `training_pipeline/eval_utils.py` | Перенесён из exp-ветки (unified evaluator) |
| `llm/prompt_builder.py` | +GNN grounding guardrails в system prompt; +key_metrics/affected_counts/gnn_provenance в user prompt |
| `remediation/pipeline.py` | +sop_chunks titles в metadata["knowledge"] для отображения в отчёте |
| `.gitignore` | +Large dataset/raw/*.pt guards; +gnn/checkpoints/*.pt; demo_data/ не игнорируется |

---

## 11. Ограничения текущей реализации

| Ограничение | Описание |
|-------------|---------|
| Синтетический датасет | Модель обучена на синтетических графах; в production нужна дообучка на реальной телеметрии |
| `fault_type` из ground truth | В demo `fault_type_hint` = ground-truth из графа. В production = отдельный fault classifier |
| `node_id` без CMDB | Индекс внутри типа (cpu#23); в production — CMDB lookup по hostname |
| Скоринг не калиброван | `confidence = sigmoid(logit)` — компаративный ранг внутри графа, не вероятность |
| Affected nodes = топология | Не ground-truth жертвы, а структурные соседи RC-узла |
| checkpoint = screening | `best_v5a_40_screening.pt` — результат short screening (10 epochs); full training (60 ep.) ожидается ≥87.5% |
| LLM без streaming | Синхронный HTTP вызов; нет streaming или async |
| Redis без персистентности | In-memory fakeredis; узловые контексты хранятся только в памяти |
| memory_bw_saturation | -2.2pp регрессия в screening (63.8%); мониторировать в full training |

---

## 12. Следующие шаги для production

1. **Full training**: запустить `python -m training_pipeline.train --epochs 60 --patience 8 --loss_mode global_ce --scorer shared --raw_dir dataset/v5a_40/raw --run_name v5a_40_full_train`
2. **Fault classifier**: отдельная модель (или эвристика по топ-1 RC-type) для получения `fault_type` без ground-truth
3. **CMDB adapter**: заменить `human_node_id()` на реальный lookup hostname/IP из CMDB
4. **Калибровка**: Platt scaling или temperature scaling для превращения логитов в P(rc_correct)
5. **Streaming LLM**: async Ollama stream + websocket для low-latency UX
6. **Персистентный Redis**: seed с реальными узловыми контекстами из инвентаря кластера
7. **Реальный Qdrant**: загрузить актуальные SOPы Говоруна + накопленные тикеты
8. **GNN monitoring**: A/B compare vs. будущих версий датасета; drift detection
9. **v5b dataset**: улучшить memory_bandwidth_saturation SNR (см. `reports/v5_feature_schema_review.md`)

---

## 13. Приложение: результаты трёх demo-прогонов

### network_congestion (data_3.pt)

```
#1  S3       (switch) score=1.0000  <== predicted root cause  CORRECT
Anomalous metrics:
    switch_packet_loss_percent   +6.04 sigma
    switch_latency_ms            +5.24 sigma
    switch_bandwidth_usage_percent +4.97 sigma
Affected: cpu=22, gpu=22, switch=2
Actions: CHECK_METRICS -> S3 | APPLY_QOS -> S3 | NOTIFY_OPERATOR
Firewall: PASSED  TTR: <5s
```

### hdd_degradation (data_11.pt)

```
#1  HDD-22   (hdd)    score=1.0000  <== predicted root cause  CORRECT
Anomalous metrics:
    disk_latency_ms              +7.67 sigma
    disk_reallocated_sectors     +7.51 sigma
    disk_read_iops               -2.42 sigma
Affected: cpu=1
Actions: CHECK_METRICS | MIGRATE_JOB | SCHEDULE_MAINTENANCE | NOTIFY_OPERATOR
Firewall: PASSED  TTR: <5s
```

### ram_leak (data_2.pt)

```
#1  RAM-22   (ram)    score=1.0000  <== predicted root cause  CORRECT
Anomalous metrics:
    ram_fragmentation_score      +5.55 sigma
    ram_available_mb             -4.67 sigma
    ram_used_percent             +3.70 sigma
    ram_page_faults_ps           +2.72 sigma
Affected: cpu=1
Actions: CHECK_METRICS | RESTART_SERVICE (checkpoint_restart) | MIGRATE_JOB | NOTIFY_OPERATOR
Firewall: PASSED  TTR: <5s
```

**Все три прогона: Hit@1 = 100%, Firewall = PASSED, TTR < 5s.**
