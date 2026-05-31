# Руководство по развёртыванию

Актуально для репозитория. Полный стек:
**edge-agent → Kafka → snapshot_engine → GNN → RAG/LLM**.

---

## Режимы запуска

| Режим | Когда использовать | Docker | Сервисы |
|---|---|---|---|
| **Offline demo** | Разработка, CI, дипломная demo | не нужен | нет |
| **Local + services** | Real LLM/RAG на хосте | опционально | Ollama/Qdrant/Redis локально |
| **Docker Compose (core)** | Воспроизводимое окружение RAG/LLM | да | redis, qdrant, ollama, jinr |
| **Docker Compose (full)** | + telemetry + Grafana | да | + kafka, jinr_api, grafana |

---

## Быстрый старт без Docker (рекомендуется для первого знакомства)

```bash
cd JINR-rag

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements_rag.txt

# Положить GNN checkpoint (не в git)
# gnn/checkpoints/best_v5a_40_screening.pt

# Тесты (23 passed)
python -m pytest tests/ -q

# E2E demo полностью offline
python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt --mock

# Только GNN
python -m gnn.inference --sample demo_data/gnn_samples/data_3.pt --top-k 5
```

Переменные окружения для GNN:

```bash
# Windows PowerShell
$env:GNN_CHECKPOINT = "gnn/checkpoints/best_v5a_40_screening.pt"
```

---

## Требования для Docker

| Ресурс | Минимум | Рекомендуется |
|---|---|---|
| Docker Desktop | 4.0+ | latest |
| Docker Compose | 2.0+ | plugin |
| Disk | 15 GB | 25 GB (Ollama models ~4 GB) |
| RAM | 8 GB | 16 GB (Mistral 7B) |
| CPU | 4 cores | 8 cores |

---

## Docker Compose: core stack

### 1. Подготовка

```bash
cd JINR-rag
cp .env.example .env   # опционально, для кастомизации
```

### 2. Инициализация

**Linux/macOS:**

```bash
chmod +x init-system.sh
./init-system.sh
```

**Windows PowerShell:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\init-system.ps1
```

**Вручную:**

```bash
docker-compose up -d qdrant redis ollama
# дождаться healthcheck (~30 с)
docker-compose exec ollama ollama pull mistral
docker-compose build jinr
docker-compose up -d jinr
```

### 3. Запуск pipeline

```bash
# RAG/LLM из inference_sample.json (default CMD контейнера jinr)
docker-compose exec jinr python -m remediation.run

# Или E2E с GNN (нужен checkpoint в volume)
docker-compose exec jinr python -m app.demo_gnn_llm_pipeline \
  --sample demo_data/gnn_samples/data_3.pt --mock
```

---

## Сервисы в `docker-compose.yml`

### Core (RAG/LLM remediation)

| Service | Container | Port | Image |
|---|---|---|---|
| redis | jinr_redis | 6379 | redis:7-alpine |
| qdrant | jinr_qdrant | 6333, 6334 | qdrant/qdrant:latest |
| ollama | jinr_ollama | 11434 | ollama/ollama:latest |
| jinr | jinr_app | 8000 | custom (Dockerfile) |

Volumes `jinr`: `./artifacts`, `./checkpoints`, `./dataset`.

### Observability (optional)

| Service | Container | Port | Назначение |
|---|---|---|---|
| jinr_api | jinr_api | 8080 | FastAPI для Grafana Infinity |
| grafana | jinr_grafana | 3000 | Node Graph dashboard (admin/jinr2024) |

```bash
docker-compose up -d jinr_api grafana
# Grafana: http://localhost:3000
# API health: curl http://localhost:8080/health
```

### Telemetry (L1–L3)

| Service | Container | Port | Назначение |
|---|---|---|---|
| kafka | jinr_kafka | 9092 | Bus для `edge-agent` → `snapshot_engine` |

Запуск consumer: `python -m snapshot_engine.run`. Edge-agent: `edge-agent/deploy/`.

---

## Архитектура Docker-сети

```text
┌─────────────────────────────────────────────────────────┐
│  Network: jinr_network                                    │
└─────────────────────────────────────────────────────────┘
         │
    ┌────┼────────┬──────────┬──────────────┐
    ▼    ▼        ▼          ▼              ▼
  Redis Qdrant  Ollama    jinr_app     jinr_api ──► Grafana
  :6379 :6333   :11434    (Python)     :8080       :3000
    │    │        │          │
    └────┴────────┴──────────┘
              │
    Shared host volumes:
    ./artifacts  ./checkpoints  ./dataset
```

---

## Конфигурация

### `.env` (из `.env.example`)

Ключевые переменные:

```bash
QDRANT_HOST=qdrant
REDIS_HOST=redis
OLLAMA_HOST=http://ollama:11434
LLM_MODEL=mistral

# Пути внутри контейнера
INFERENCE_PATH=/app/artifacts/inference_sample.json
REPORT_PATH=/app/artifacts/remediation_report.json
```

Для GNN checkpoint вне compose defaults:

```bash
GNN_CHECKPOINT=/app/gnn/checkpoints/best_v5a_40_screening.pt
```

Смонтируйте checkpoint:

```yaml
# docker-compose.override.yml (локально)
services:
  jinr:
    volumes:
      - ./gnn/checkpoints:/app/gnn/checkpoints:ro
```

### Файлы конфигурации

| Файл | Назначение |
|---|---|
| `docker-compose.yml` | Определение сервисов |
| `.env.example` | Шаблон переменных |
| `Dockerfile` | Python 3.12 app image |
| `init-system.sh` / `init-system.ps1` | Авто-init |

---

## Операции

### Старт / стоп

```bash
docker-compose up -d
docker-compose logs -f jinr

docker-compose down          # сохранить volumes
docker-compose down -v       # удалить volumes
```

### Команды в контейнере

```bash
docker-compose exec jinr python -m remediation.run
docker-compose exec jinr python -m app.demo_gnn_llm_pipeline \
  --sample demo_data/gnn_samples/data_3.pt --mock
docker-compose exec jinr pytest tests/ -v
docker-compose exec jinr /bin/bash
```

### Health checks

```bash
docker-compose ps

curl http://localhost:6333/health       # Qdrant
curl http://localhost:11434/api/tags    # Ollama
docker exec jinr_redis redis-cli ping   # Redis
curl http://localhost:8080/health        # Grafana API
```

### Rebuild после изменений кода

```bash
docker-compose build jinr
docker-compose up -d jinr
```

---

## Troubleshooting

### Docker daemon не запущен

- Windows/macOS: открыть Docker Desktop
- Linux: `sudo systemctl start docker`

### Порт занят (6333, 11434, 3000, …)

```powershell
# Windows
netstat -ano | findstr :6333
```

Изменить mapping в `docker-compose.yml` или остановить конфликтующий процесс.

### OOM при Mistral 7B

1. Увеличить RAM для Docker Desktop (Settings → Resources)
2. Quantized model: `ollama pull mistral:4bit`, `LLM_MODEL=mistral:4bit`
3. Demo без LLM: `--mock-llm`

### Qdrant collection not found

```bash
docker-compose exec jinr python -c \
  "from rag.knowledge_base import load_sops; load_sops()"
```

### GNN checkpoint missing

```text
FileNotFoundError: gnn/checkpoints/best_v5a_40_screening.pt
```

Checkpoint не в git. Скопируйте обученную модель в `gnn/checkpoints/` или задайте
`GNN_CHECKPOINT`. Для RAG-only demo checkpoint не нужен — используйте
`python -m remediation.run`.

---

## CI / testing

```bash
# Local
python -m pytest tests/ -q

# Docker
docker-compose up -d qdrant redis
docker-compose build jinr
docker-compose exec -T jinr pytest tests/ -q
docker-compose down
```

Mock mode не требует Ollama/Qdrant и подходит для CI.

---

## GPU acceleration (Ollama)

На Linux с NVIDIA GPU можно добавить в `docker-compose.yml`:

```yaml
ollama:
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
```

Требуется nvidia-container-toolkit.

---

## Backup volumes

```bash
# Qdrant
docker run --rm -v jinr-rag_qdrant_storage:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/qdrant-backup.tar.gz -C /data .

# Redis
docker run --rm -v jinr-rag_redis_storage:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/redis-backup.tar.gz -C /data .

# Ollama models
docker run --rm -v jinr-rag_ollama_models:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/ollama-backup.tar.gz -C /data .
```

---

## Cleanup

```bash
docker-compose down -v
docker system prune -a --volumes   # осторожно: удалит все unused images
```

---

## Связанные документы

| Документ | Содержание |
|---|---|
| [`README.md`](README.md) | Обзор, benchmarks, команды |
| [`LAYERS.md`](LAYERS.md) | Послойная модель pipeline |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Software + domain architecture |
| [`docs/LEGACY_ARCHIVE.md`](docs/LEGACY_ARCHIVE.md) | Архив legacy-компонентов |

---

## References

- [Docker Compose](https://docs.docker.com/compose)
- [Qdrant docs](https://qdrant.tech/documentation)
- [Ollama](https://github.com/ollama/ollama)
- [Redis docs](https://redis.io/documentation)
