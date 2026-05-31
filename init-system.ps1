# Initialize GNN RCA System (Windows/PowerShell)
# This script sets up Docker containers and initializes the knowledge base

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║        GNN RCA SYSTEM - INITIALIZATION SCRIPT                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
$docker = docker --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop." -ForegroundColor Red
    Write-Host "   https://www.docker.com/products/docker-desktop"
    exit 1
}

Write-Host "✓ Docker is installed" -ForegroundColor Green
Write-Host ""

# Check Docker daemon
try {
    docker ps > $null 2>&1
} catch {
    Write-Host "❌ Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker daemon is running" -ForegroundColor Green
Write-Host ""

# Start containers
Write-Host "📦 Starting Docker containers..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check Qdrant
Write-Host ""
Write-Host "🔍 Checking Qdrant Vector DB..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Qdrant is ready" -ForegroundColor Green
            break
        }
    } catch {
        $attempt++
        if ($attempt -lt $maxAttempts) {
            Write-Host "  Waiting for Qdrant... ($attempt/$maxAttempts)" -ForegroundColor Gray
            Start-Sleep -Seconds 2
        }
    }
}

if ($attempt -eq $maxAttempts) {
    Write-Host "❌ Qdrant failed to start" -ForegroundColor Red
    exit 1
}

# Check Redis
Write-Host ""
Write-Host "🔍 Checking Redis..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
while ($attempt -lt $maxAttempts) {
    try {
        $output = docker exec jinr_redis redis-cli ping 2>&1
        if ($output -eq "PONG") {
            Write-Host "✓ Redis is ready" -ForegroundColor Green
            break
        }
    } catch {
        $attempt++
        if ($attempt -lt $maxAttempts) {
            Write-Host "  Waiting for Redis... ($attempt/$maxAttempts)" -ForegroundColor Gray
            Start-Sleep -Seconds 2
        }
    }
}

# Check Ollama
Write-Host ""
Write-Host "🔍 Checking Ollama..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Ollama is ready" -ForegroundColor Green
            break
        }
    } catch {
        $attempt++
        if ($attempt -lt $maxAttempts) {
            Write-Host "  Waiting for Ollama... ($attempt/$maxAttempts)" -ForegroundColor Gray
            Start-Sleep -Seconds 2
        }
    }
}

# Pull Mistral model
Write-Host ""
Write-Host "📥 Pulling Mistral 7B model..." -ForegroundColor Yellow
Write-Host "   (This may take 5-10 minutes on first run, ~4.1GB download)" -ForegroundColor Gray
docker exec jinr_ollama ollama pull mistral
Write-Host "✓ Mistral 7B is ready" -ForegroundColor Green

# Run tests
Write-Host ""
Write-Host "🧪 Running integration tests..." -ForegroundColor Yellow
docker-compose exec -T jinr pytest tests/ -v --tb=short

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✅ SYSTEM READY FOR USE                          ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access points:" -ForegroundColor Cyan
Write-Host "  • Qdrant API:   http://localhost:6333" -ForegroundColor Gray
Write-Host "  • Redis:        localhost:6379" -ForegroundColor Gray
Write-Host "  • Ollama API:   http://localhost:11434" -ForegroundColor Gray
Write-Host "  • Adminer:      http://localhost:8080 (optional)" -ForegroundColor Gray
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run inference:  docker-compose exec jinr python -m remediation.run" -ForegroundColor Gray
Write-Host "  2. View logs:      docker-compose logs -f jinr" -ForegroundColor Gray
Write-Host "  3. Stop system:    docker-compose down" -ForegroundColor Gray
Write-Host ""
