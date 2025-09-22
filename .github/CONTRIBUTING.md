# Contributing to Satellite Anomaly Flood-Fill

Thank you for your interest in contributing to this satellite anomaly detection system!

## Development Process

1. Fork the repository
2. Create a feature branch from `develop`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass and clippy warnings are resolved
6. Update documentation as needed
7. Submit a pull request

## Coding Standards

- Follow Rust naming conventions (snake_case for variables/functions, PascalCase for types)
- Use `no_std` for flight code, `std` for simulation code
- Add comprehensive error handling with `thiserror`
- Use `heapless` for deterministic memory allocation
- Include detailed documentation comments for all public APIs
- Implement proper bounds checking and memory safety

## Testing Requirements

- Write unit tests for all new functions
- Add integration tests for complete workflows
- Include property-based tests for edge cases
- Validate performance benchmarks
- Test memory usage patterns

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Link any relevant issues
3. Include test results and benchmark data if applicable
4. Update the CHANGELOG.md if your changes are user-facing
5. Ensure CI passes before requesting review

## Code Review

- All submissions require review by project maintainers
- Focus on correctness, performance, and safety
- Consider real-time constraints and embedded system limitations
- Validate adherence to space systems best practices

## Questions?

Feel free to open an issue for discussion before starting work on large features.
