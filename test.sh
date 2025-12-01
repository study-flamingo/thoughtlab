#!/bin/bash
#
# test.sh - Run all tests for ThoughtLab
#
# Usage:
#   ./test.sh           # Run all tests
#   ./test.sh backend   # Run only backend tests
#   ./test.sh frontend  # Run only frontend tests
#   ./test.sh --quick   # Run only unit tests (no Neo4j required)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results
BACKEND_RESULT=0
FRONTEND_RESULT=0

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

run_backend_tests() {
    print_header "Running Backend Tests (pytest)"
    
    cd backend
    
    # Activate virtual environment
    if [[ -f ".venv/Scripts/activate" ]]; then
        source .venv/Scripts/activate
    elif [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    else
        print_error "Backend virtual environment not found!"
        print_warning "Run: cd backend && uv venv && uv pip install -r requirements.txt"
        cd ..
        return 1
    fi
    
    # Run tests based on mode
    if [[ "$QUICK_MODE" == "true" ]]; then
        echo "Running quick tests (models only, no Neo4j required)..."
        python -m pytest tests/test_activity_models.py tests/test_models.py -v --tb=short
    else
        echo "Running all backend tests..."
        python -m pytest tests/ -v --tb=short
    fi
    
    local result=$?
    cd ..
    return $result
}

run_frontend_tests() {
    print_header "Running Frontend Tests (vitest)"
    
    cd frontend
    
    # Check if node_modules exists
    if [[ ! -d "node_modules" ]]; then
        print_error "Frontend dependencies not installed!"
        print_warning "Run: cd frontend && npm install"
        cd ..
        return 1
    fi
    
    # Run tests
    npm test -- --run
    
    local result=$?
    cd ..
    return $result
}

print_summary() {
    print_header "Test Summary"
    
    if [[ $BACKEND_RESULT -eq 0 ]]; then
        print_success "Backend tests passed"
    else
        print_error "Backend tests failed"
    fi
    
    if [[ $FRONTEND_RESULT -eq 0 ]]; then
        print_success "Frontend tests passed"
    else
        print_error "Frontend tests failed"
    fi
    
    echo ""
    
    if [[ $BACKEND_RESULT -eq 0 && $FRONTEND_RESULT -eq 0 ]]; then
        echo -e "${GREEN}All tests passed! ✓${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed. See output above for details.${NC}"
        return 1
    fi
}

# Parse arguments
QUICK_MODE="false"
RUN_BACKEND="true"
RUN_FRONTEND="true"

for arg in "$@"; do
    case $arg in
        backend)
            RUN_FRONTEND="false"
            ;;
        frontend)
            RUN_BACKEND="false"
            ;;
        --quick|-q)
            QUICK_MODE="true"
            ;;
        --help|-h)
            echo "Usage: ./test.sh [backend|frontend] [--quick]"
            echo ""
            echo "Options:"
            echo "  backend   Run only backend tests"
            echo "  frontend  Run only frontend tests"
            echo "  --quick   Run only unit tests (no Neo4j required)"
            echo "  --help    Show this help message"
            exit 0
            ;;
    esac
done

# Main execution
echo -e "${BLUE}ThoughtLab Test Runner${NC}"
echo "======================================"

if [[ "$RUN_BACKEND" == "true" ]]; then
    if run_backend_tests; then
        BACKEND_RESULT=0
    else
        BACKEND_RESULT=1
    fi
else
    echo "Skipping backend tests"
fi

if [[ "$RUN_FRONTEND" == "true" ]]; then
    if run_frontend_tests; then
        FRONTEND_RESULT=0
    else
        FRONTEND_RESULT=1
    fi
else
    echo "Skipping frontend tests"
fi

print_summary

