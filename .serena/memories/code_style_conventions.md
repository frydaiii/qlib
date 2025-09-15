# Code Style and Conventions

## Python Code Style
- Follow PEP 8 guidelines
- Use Black for automatic code formatting
- Line length: 120 characters (as configured in .pylintrc)
- Use type hints throughout the codebase
- Import order: standard library, third-party, local imports

## Naming Conventions
- Classes: PascalCase (e.g., `VNStockCollector`, `DumpDataUpdate`)
- Functions/methods: snake_case (e.g., `update_data_to_bin`, `get_vn_stock_symbols`)
- Constants: UPPER_SNAKE_CASE (e.g., `CUR_DIR`, `DAILY_FORMAT`)
- Private methods: prefix with underscore (e.g., `_get_old_data`, `_dump_bin`)

## Documentation Style
- Use Google-style docstrings with Parameters and Returns sections
- Include Examples section for CLI commands
- Document complex algorithms and business logic
- Type hints in function signatures supplement docstring documentation

## File Structure
- Base classes in `base.py` files
- Region-specific implementations in dedicated directories
- Abstract methods clearly marked with `@abc.abstractmethod`
- Import statements follow: standard lib, third-party, qlib internals, local

## Error Handling
- Use specific exception types where appropriate
- Log warnings and errors using loguru logger
- Validate parameters at function entry points
- Graceful degradation with fallback mechanisms

## Testing Conventions
- Test files prefixed with `test_`
- Use pytest framework with fixtures
- Mock external dependencies
- Separate unit tests from integration tests