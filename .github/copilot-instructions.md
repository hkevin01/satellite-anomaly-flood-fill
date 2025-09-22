# Copilot Instructions for Satellite Anomaly Flood-Fill

## Project Overview
This is a Rust workspace project for satellite onboard anomaly isolation using flood-fill algorithms on fault maps. The project is designed for space systems fault detection, isolation, and recovery (FDIR) with real-time constraints.

## Code Standards
- Use `no_std` for flight code (`flight_app` crate)
- Use `std` for simulation code (`sim_host` crate)
- Follow Rust naming conventions (snake_case for variables/functions, PascalCase for types)
- Add comprehensive error handling with `thiserror` for error types
- Use `heapless` for deterministic memory allocation
- Include detailed documentation comments for all public APIs
- Implement proper bounds checking and memory safety
- Add time measurement and performance metrics
- Handle graceful degradation under failure conditions

## Architecture
- `anomaly_map`: Data structures for grid representations
- `floodfill_core`: Core flood-fill algorithms
- `features`: Region analysis and feature extraction
- `decisions`: Policy engine for GN&C/FDIR actions
- `flight_app`: Flight software implementation (no_std)
- `sim_host`: Host simulation and testing (std)

## Testing Requirements
- Unit tests for all algorithms
- Integration tests for complete workflows
- Property-based testing for edge cases
- Performance benchmarks
- Memory usage validation

## Performance Constraints
- Deterministic execution time
- Fixed memory allocation patterns
- Real-time operation capability
- Graceful error handling without panics
