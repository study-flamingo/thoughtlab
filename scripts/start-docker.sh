#!/bin/bash
# ThoughtLab - Start Containerized Development
# Starts all services (frontend, backend, Neo4j, Redis) in Docker containers

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 ThoughtLab - Containerized Development"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${BLUE}Starting all services in Docker containers...${NC}"
echo ""

# Stop any existing containers first
echo "🧹 Cleaning up existing containers..."
docker-compose down

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Wait for frontend health check specifically
echo "🔍 Checking frontend health..."
for i in {1..15}; do
    if curl -f http://localhost:5173 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Frontend is healthy"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${YELLOW}⚠ Frontend health check still in progress${NC}"
        echo -e "${YELLOW}   (Container is running, but Docker health checks may show 'unhealthy' temporarily)${NC}"
    fi
    sleep 2
done

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Test backend health
echo ""
echo "🔍 Testing backend health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend is healthy"
else
    echo -e "${YELLOW}⚠ Backend health check failed, but service may still be starting${NC}"
fi

# Test frontend connectivity
echo ""
echo "🔍 Testing frontend connectivity..."
if curl -f http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Frontend is accessible"
else
    echo -e "${YELLOW}⚠ Frontend connectivity check failed${NC}"
fi

# Show summary
echo ""
echo "═══════════════════════════════════════════="
echo "✅ Containerized Development Ready!"
echo "═══════════════════════════════════════════="
echo ""
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  Neo4j:    http://localhost:7474"
echo "  Redis:    localhost:6379"
echo ""
echo "  View Logs:     docker-compose logs -f"
echo "  Stop Services: docker-compose down"
echo "  Restart:       docker-compose restart [service]"
echo ""
echo "  Frontend uses: http://backend:8000/api/v1 (internal Docker network)"
echo "═══════════════════════════════════════════="
echo ""

# Save mode marker
touch ".containerized-frontend"

# Offer to tail logs
read -p "Do you want to view logs in real-time? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Viewing logs (Ctrl+C to stop):"
    echo "═══════════════════════════════════════════="
    docker-compose logs -f
else
    echo "💡 Tip: Run 'docker-compose logs -f' to view logs"
    echo "💡 Tip: Run 'docker-compose down' to stop services"
fi