#!/bin/bash

# Satellite Anomaly Flood-Fill Build Script
# Comprehensive build and validation pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="satellite-anomaly-flood-fill"
BUILD_TYPE="${BUILD_TYPE:-release}"
TARGET_ARCH="${TARGET_ARCH:-x86_64-unknown-linux-gnu}"
CARGO_FLAGS="${CARGO_FLAGS:-}"

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

check_dependencies() {
    log_info "Checking build dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        log_warning "Git not found. Version information may be incomplete."
    fi
    
    log_success "Dependencies check passed"
}

clean_build() {
    log_info "Cleaning previous build artifacts..."
    cargo clean
    log_success "Clean completed"
}

validate_code() {
    log_info "Running code validation..."
    
    # Format check
    log_info "Checking code formatting..."
    if ! cargo fmt --all -- --check; then
        log_error "Code formatting issues found. Run 'cargo fmt' to fix."
        exit 1
    fi
    
    # Clippy linting
    log_info "Running Clippy linting..."
    cargo clippy --all-targets --all-features -- -D warnings
    
    log_success "Code validation passed"
}

build_workspace() {
    log_info "Building workspace in ${BUILD_TYPE} mode..."
    
    local build_flags=""
    if [[ "${BUILD_TYPE}" == "release" ]]; then
        build_flags="--release"
    fi
    
    # Build all crates
    cargo build ${build_flags} ${CARGO_FLAGS} --target ${TARGET_ARCH}
    
    log_success "Workspace build completed"
}

print_summary() {
    log_success "Build Summary"
    echo "=============="
    echo "Project: ${PROJECT_NAME}"
    echo "Build Type: ${BUILD_TYPE}"
    echo "Target: ${TARGET_ARCH}"
    echo "Timestamp: $(date)"
    
    if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        echo "Git Commit: $(git rev-parse --short HEAD)"
        echo "Git Branch: $(git branch --show-current)"
    fi
    
    echo ""
    log_success "Build completed successfully! 🚀"
}

# Main execution
main() {
    log_info "Starting ${PROJECT_NAME} build pipeline..."
    
    check_dependencies
    build_workspace
    print_summary
}

# Run main function with all arguments
main "$@"
