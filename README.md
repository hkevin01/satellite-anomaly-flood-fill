# 🛰️ Satellite Anomaly Flood-Fill

[![Build Status](https://github.com/hkevin01/satellite-anomaly-flood-fill/workflows/CI/badge.svg)](https://github.com/hkevin01/satellite-anomaly-flood-fill/actions)
[![Coverage](https://codecov.io/gh/hkevin01/satellite-anomaly-flood-fill/branch/main/graph/badge.svg)](https://codecov.io/gh/hkevin01/satellite-anomaly-flood-fill)
[![Rust Version](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced satellite onboard anomaly isolation using flood-fill algorithms for space systems fault detection, isolation, and recovery (FDIR)**

## �� Overview

This project implements a sophisticated flood-fill based anomaly detection system designed for small satellites and deep-space probes. The system can quickly isolate contiguous regions of anomalies on 2D/3D grids representing various spacecraft subsystems (solar arrays, radiators, sensor planes) and trigger appropriate containment actions.

### Key Features

- **🚀 Space-Grade Software**: Designed for real-time embedded systems with `no_std` support
- **⚡ High Performance**: Deterministic execution time with bounded memory allocation
- **🛡️ Robust Error Handling**: Comprehensive error management with graceful degradation
- **📊 Advanced Analytics**: Real-time anomaly growth tracking and threat assessment
- **🔧 Flexible Architecture**: Modular design supporting various grid types and connectivity patterns
- **🧪 Extensive Testing**: Property-based testing, fuzzing, and performance benchmarks

## 🏗️ Architecture

The project is organized as a Rust workspace with multiple specialized crates:

```
satellite-anomaly-flood-fill/
├── src/
│   ├── anomaly_map/      # Grid data structures and cell management
│   ├── floodfill_core/   # Core flood-fill algorithms (4-conn, 8-conn)
│   ├── features/         # Region analysis and temporal tracking
│   ├── decisions/        # Policy engine for GN&C/FDIR actions
│   ├── flight_app/       # Flight software implementation (no_std)
│   └── sim_host/         # Host simulation and testing (std)
├── tests/                # Integration and property-based tests
├── benches/              # Performance benchmarks
├── docs/                 # Comprehensive documentation
├── scripts/              # Build and deployment scripts
└── data/                 # Test data and configurations
```

## 🚀 Quick Start

### Prerequisites

- **Rust**: Version 1.70 or later ([Install Rust](https://rustup.rs/))
- **Git**: For version control
- **Optional**: `cargo-tarpaulin` for coverage, `cargo-criterion` for benchmarks

### Installation

```bash
# Clone the repository
git clone https://github.com/hkevin01/satellite-anomaly-flood-fill.git
cd satellite-anomaly-flood-fill

# Build the project
./scripts/build.sh

# Run tests
./scripts/test.sh

# Run the simulation
cargo run --bin sim_host
```

### Basic Usage

```rust
use satellite_anomaly_flood_fill::*;

// Create a 64x32 solar panel grid
let mut panel = Grid2D::new(64, 32, Cell::default())?;

// Simulate thermal anomaly
panel.get_mut(10, 15)?.update(450, 1200, 8, get_timestamp())?; // 45°C hot spot

// Extract anomalous regions
let components = extract_components(64, 32, 
    |x, y| matches!(panel.get(x, y)?.status, CellStatus::Anomalous),
    Connectivity::Four
)?;

// Make decisions based on detected anomalies
let policy = Policy::default();
let context = DecisionContext::default();
let action = decide(&components, &policy, &context)?;

println!("Recommended action: {}", action);
```

## 📋 Use Cases

### 1. **Solar Array Fault Management**
- Detect hot-spots and current anomalies in solar cells
- Isolate affected power strings to prevent cascade failures
- Maintain power budget optimization

### 2. **Thermal System Protection**
- Monitor radiator panel temperatures
- Detect thermal gradients and blocked cooling paths
- Trigger attitude adjustments for thermal protection

### 3. **Star Tracker Anomaly Handling**
- Identify hot pixels and radiation damage
- Mask corrupted sensor regions
- Maintain attitude determination accuracy

### 4. **Propulsion System Monitoring**
- Track thruster performance anomalies
- Detect fuel line blockages or leaks
- Enable backup system activation

## 🎯 Performance Characteristics

### Real-Time Performance
- **Grid Processing**: < 2ms for 64x32 grids
- **Decision Making**: < 1ms response time
- **Memory Usage**: < 1MB for typical scenarios
- **Stack Usage**: Bounded to 4KB maximum

### Scalability
- Supports grids up to 512x512 cells
- Handles 100+ simultaneous anomalous regions
- Deterministic execution regardless of anomaly complexity

### Resource Efficiency
- Zero-allocation flood-fill using stack-based approach
- Optional `no_std` support for embedded deployment
- Configurable memory limits for different platforms

## 🧪 Testing & Validation

The project includes comprehensive testing at multiple levels:

### Test Categories
- **Unit Tests**: Individual component validation
- **Integration Tests**: Cross-component interaction testing
- **Property Tests**: Invariant checking with `proptest`
- **Stress Tests**: Large-scale grid operations
- **Memory Tests**: Leak detection and bounds checking
- **Performance Tests**: Timing and throughput validation

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run with coverage
./scripts/test.sh --coverage

# Run only unit tests
./scripts/test.sh --unit-only

# Run stress tests
./scripts/test.sh --no-memory  # Skip memory-intensive tests
```

## 🛠️ Development

### Building for Different Targets

```bash
# Standard build with std library
cargo build --release

# Embedded/flight build (no_std)
cargo build --release --no-default-features --features no_std

# Cross-compilation for ARM Cortex-M
cargo build --target thumbv7em-none-eabihf --no-default-features --features no_std
```

## 📚 Documentation

- **[Project Plan](docs/project_plan.md)**: Detailed implementation phases and milestones
- **[API Documentation](target/doc/)**: Generated from source code comments
- **[Architecture Guide](docs/architecture.md)**: System design and component interactions
- **[Performance Guide](docs/performance.md)**: Optimization and benchmarking details

## 🌍 Real-World Applications

This system is designed for deployment in:

- **CubeSats**: Small satellite missions with limited computational resources
- **Deep Space Probes**: Long-duration missions requiring autonomous fault management
- **Commercial Satellites**: Constellation missions needing rapid anomaly response
- **Space Stations**: Complex systems with multiple subsystem interactions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready for space deployment! 🚀**
