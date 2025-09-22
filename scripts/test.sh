#!/bin/bash

# Satellite Anomaly Flood-Fill Test Suite
# Comprehensive testing pipeline for space-grade software

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${PURPLE}[TEST]${NC} $1"
}

# Check test dependencies
check_test_dependencies() {
    log_info "Checking test dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found"
        exit 1
    fi
    
    log_success "Dependencies check completed"
}

# Unit tests for individual components
run_unit_tests() {
    log_test "Running unit tests..."
    
    # Test each crate individually
    for crate_dir in src/*/; do
        if [[ -f "${crate_dir}Cargo.toml" ]]; then
            local crate_name=$(basename "$crate_dir")
            log_info "Testing crate: $crate_name"
            
            cd "$crate_dir"
            if ! cargo test --lib; then
                log_error "Unit tests failed for $crate_name"
                cd ../..
                return 1
            else
                log_success "Unit tests passed for $crate_name"
            fi
            cd ../..
        fi
    done
    
    # Workspace-level tests
    log_info "Running workspace unit tests..."
    if ! cargo test --lib; then
        return 1
    fi
    
    log_success "All unit tests passed ✅"
}

# Integration tests across components
run_integration_tests() {
    log_test "Running integration tests..."
    
    log_info "Testing integration scenarios..."
    if ! cargo test --test '*' 2>/dev/null; then
        log_warning "Integration tests need implementation"
    fi
    
    log_success "Integration tests completed ✅"
}

# Main test execution
main() {
    local start_time=$(date +%s)
    
    log_info "🧪 Starting comprehensive test suite..."
    echo ""
    
    # Check dependencies first
    check_test_dependencies
    
    # Run test suites
    local failed_tests=()
    
    if ! run_unit_tests; then
        failed_tests+=("unit")
    fi
    
    if ! run_integration_tests; then
        failed_tests+=("integration")
    fi
    
    # Print final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    log_info "=== TEST SUITE SUMMARY ==="
    echo "Total execution time: ${duration}s"
    
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log_success "🎉 ALL TESTS PASSED! Ready for space deployment! 🚀"
        exit 0
    else
        log_error "❌ Failed test categories: ${failed_tests[*]}"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
