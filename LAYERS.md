# Послойная модель pipeline

Документ описывает **актуальный** end-to-end pipeline: L1–L3 telemetry,
L4 GNN, L5–L6 RAG/LLM. Старые phase-артефакты — в
[`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md).

---

## Обзор слоёв

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 1–2: TELEMETRY (edge-agent, Go)                        ┃
┃  LAYER 3: SNAPSHOT (snapshot_engine, Kafka + Polars)          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 4 / GNN: RCA (graph → ranked root cause)               ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Entry:  demo_data/gnn_samples/*.pt  или  production graph    ┃
┃  Code:   gnn/inference.py, gnn/model.py                       ┃
┃  Output: top-k RC candidates, logits, anomalous metrics        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 0: INCIDENT ADAPTER (GNN → inference contract)         ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Code:   integrations/gnn_to_incident.py                      ┃
┃  Output: { graph_id, rc_node, confidence, top5_candidates,    ┃
┃            victim_nodes, fault_type, key_metrics, ... }       ┃
┃  Alt input: artifacts/inference_sample.json (без GNN)         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 1: KNOWLEDGE RETRIEVAL (RAG)                           ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Code:   rag/retriever.py, rag/qdrant_store.py,               ┃
┃          rag/redis_context.py, rag/knowledge_base.py,         ┃
┃          rag/history_tickets.py                                 ┃
┃  Services (optional): Qdrant :6333, Redis :6379                 ┃
┃  Fallback: empty context / mock SOP (--mock-rag)              ┃
┃  Output: sop_chunks, retrieval_method, incident history       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 2: REASONING & GENERATION (LLM)                        ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Code:   llm/llm_client.py, llm/prompt_builder.py,            ┃
┃          llm/response_parser.py                               ┃
┃  Backend: Ollama (mistral) or rule-based fallback             ┃
┃  Output: structured playbook { actions: [...] }               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 3: VALIDATION & SECURITY (FIREWALL)                    ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Code:   remediation/firewall.py, remediation/models.py       ┃
┃  Rules:  whitelist Action DSL, keyword blacklist              ┃
┃  Output: firewall_status PASSED | BLOCKED                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              v
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LAYER 4: AGGREGATION & OUTPUT (REPORT)                       ┃
┃  ───────────────────────────────────────────────────────────  ┃
┃  Code:   remediation/incident_aggregator.py,                  ┃
┃          remediation/pipeline.py, remediation/run.py          ┃
┃  Output: artifacts/remediation_report.json                    ┃
┃          artifacts/gnn_llm_demo_trace.json (E2E demo)         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Ответственность слоёв

| Слой | Ответственность | Сбой | Деградация |
|---|---|---|---|
| -1 GNN | Ранжирование 406 RC-кандидатов | Нет checkpoint → ошибка | — |
| 0 Adapter | Нормализация GNN output в contract | Пустой top-k | null RC |
| 1 RAG | SOP + context + history | Qdrant/Redis недоступны | mock / empty context |
| 2 LLM | Генерация playbook | Ollama недоступен | rule-based fallback |
| 3 Firewall | Блокировка опасных actions | action blocked | fallback playbook |
| 4 Report | Агрегация metadata + TTR | — | всегда пишет JSON |

---

## Точки входа

### Полный pipeline: GNN → RAG → LLM (рекомендуется)

```bash
# Полностью offline (mock LLM + mock RAG)
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt --mock

# Real mode (нужны Ollama, Qdrant, Redis — или auto-fallback)
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt
```

### Только GNN inference

```bash
python -m gnn.inference --sample demo_data/gnn_samples/data_3.pt --top-k 5
```

### Только RAG/LLM (из готового inference JSON)

```bash
python -m remediation.run
# читает artifacts/inference_sample.json
# пишет  artifacts/remediation_report.json
```

### Программный вызов

```python
from gnn.inference import GNNInferenceEngine
from integrations.gnn_to_incident import gnn_to_inference
from remediation.pipeline import run_pipeline

engine = GNNInferenceEngine()
gnn_out = engine.predict("demo_data/gnn_samples/data_3.pt", top_k=5)
inference = gnn_to_inference(gnn_out)
playbook, metadata = run_pipeline(inference)
```

---

## Тесты

Актуальный test suite — **36 тестов**:

```bash
python -m pytest tests/ -v
```

| Файл | Покрытие |
|---|---|
| `tests/test_gnn_integration.py` | adapter, prompt guardrails, firewall, E2E mock, real GNN inference |
| `tests/test_rag_pipeline.py` | firewall, pipeline mock, Qdrant/fake-redis, TTR budget |
| `tests/test_full_system_integration.py` | full stack integration |
| `tests/test_diagnostics.py` | batch invariance, diagnostics |
| `tests/test_listwise.py` | listwise loss / ranking |

---

## Бюджет задержек (ориентир)

Typical run на локальной машине (mock mode):

| Этап | Latency | Доля |
|---|---:|---:|
| GNN inference | ~10–50 ms | <1% |
| RAG retrieval | <5 ms (cache) | <0.1% |
| LLM (Mistral 7B) | 3–10 s | доминирует |
| Firewall | <1 ms | <0.1% |
| Report | <10 ms | <0.1% |

В real mode LLM доминирует. Оптимизации: `--mock-llm`, quantized model, кэш Redis.

---

## Зависимости

### Python (см. `requirements.txt`, `requirements_rag.txt`)

- **torch**, **torch_geometric** — GNN
- **pydantic** — Action DSL / playbook schema
- **qdrant-client**, **redis** — RAG (optional)
- **httpx/requests** — Ollama API
- **pytest** — тests

### Внешние сервисы (optional)

| Сервис | Назначение | Offline fallback |
|---|---|---|
| Ollama :11434 | LLM inference | rule-based playbook |
| Qdrant :6333 | vector SOP search | mock / empty |
| Redis :6379 | incident context cache | in-memory skip |

Core safety layer (firewall) работает **всегда локально**, без внешних сервисов.

---

## Связанные документы

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — домен + software architecture
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker Compose и init scripts
- [`README.md`](README.md) — метрики, benchmarks, команды
- [`remediation/README.md`](remediation/README.md) — детали Action DSL
