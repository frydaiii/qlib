# Development Commands and Tools

## Essential Commands

### Building and Development
```bash
# Install in development mode
pip install -e .

# Clean build artifacts
make clean

# Build documentation
make docs

# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_model.py

# Format code with black
make black

# Run linting
make lint
make pylint
make flake8

# Type checking
make mypy
```

### Data Collection and Processing
```bash
# Download data (Vietnamese stocks example)
python scripts/data_collector/vnstock/collector.py download_data \
  --source_dir ~/.qlib/stock_data/source \
  --region VN --start 2020-11-01 --end 2020-11-10 \
  --delay 0.1 --interval 1D

# Normalize data
python scripts/data_collector/vnstock/collector.py normalize_data \
  --source_dir ~/.qlib/stock_data/source \
  --normalize_dir ~/.qlib/stock_data/normalize \
  --region VN --interval 1D

# Update existing data to binary format
python scripts/data_collector/vnstock/collector.py update_data_to_bin \
  --qlib_data_1d_dir <user data dir> \
  --trading_date <start date> --end_date <end date>

# Dump data to binary format
python scripts/dump_bin.py dump_all \
  --data_path <data_path> --qlib_dir <qlib_dir> \
  --freq day --exclude_fields symbol,date
```

### Testing and Quality Assurance
```bash
# Run full test suite
make test

# Run with coverage
pytest --cov=qlib tests/

# Check code style
make lint

# Format code
make black

# Run type checking
make mypy
```

## Git Workflow
- Follow conventional commits
- Use pre-commit hooks for automated checks
- Run full test suite before pushing