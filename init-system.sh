#!/bin/bash
# Initialize GNN RCA System
# This script sets up Docker containers and initializes the knowledge base

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        GNN RCA SYSTEM - INITIALIZATION SCRIPT                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install Docker Compose."
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Start containers
echo "📦 Starting Docker containers..."
docker-compose up -d

echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check Qdrant
echo ""
echo "🔍 Checking Qdrant Vector DB..."
until curl -s http://localhost:6333/health | grep -q "ok"; do
    echo "  Waiting for Qdrant..."
    sleep 2
done
echo "✓ Qdrant is ready"

# Check Redis
echo ""
echo "🔍 Checking Redis..."
until docker exec jinr_redis redis-cli ping &> /dev/null; do
    echo "  Waiting for Redis..."
    sleep 2
done
echo "✓ Redis is ready"

# Check Ollama
echo ""
echo "🔍 Checking Ollama..."
until curl -s http://localhost:11434/api/tags &> /dev/null; do
    echo "  Waiting for Ollama..."
    sleep 2
done
echo "✓ Ollama is ready"

# Pull Mistral model
echo ""
echo "📥 Pulling Mistral 7B model (this may take a few minutes)..."
docker exec jinr_ollama ollama pull mistral

echo ""
echo "✓ Mistral 7B is ready"

# Initialize knowledge base
echo ""
echo "📚 Initializing knowledge base..."
docker exec jinr_app python -c "
from rag.knowledge_base import load_sops
from rag.qdrant_store import QdrantStore

print('  Loading SOPs...')
load_sops()
print('  ✓ Knowledge base initialized')
"

# Run tests
echo ""
echo "🧪 Running integration tests..."
docker exec jinr_app pytest tests/ -v --tb=short

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              ✅ SYSTEM READY FOR USE                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Access points:"
echo "  • Application:  http://localhost:8000"
echo "  • Qdrant API:   http://localhost:6333"
echo "  • Redis:        localhost:6379"
echo "  • Ollama API:   http://localhost:11434"
echo "  • Adminer:      http://localhost:8080"
echo ""
echo "📋 Next steps:"
echo "  1. Run: docker-compose exec jinr python -m remediation.run"
echo "  2. View output in: docker-compose logs jinr"
echo "  3. Stop system: docker-compose down"
echo ""
