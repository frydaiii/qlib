# Repository Guidelines

## Project Structure & Module Organization
`qlib/` hosts the core Python package, including data loaders, models, and backtest utilities. `tests/` mirrors module boundaries with end-to-end and unit suites; add new scenarios beside the feature under test. Reference notebooks and runnable pipelines live in `examples/`, while CLI helpers and maintenance scripts stay under `scripts/`. Documentation sources reside in `docs/`; generated artifacts and experiment outputs should remain in `build/` and `mlruns/` and must not be versioned.

## Build, Test, and Development Commands
- `make prerequisite`: compiles the Cython extensions in `qlib/data/_libs` so rolling and expanding ops are importable.
- `make develop`: installs the dev extras (`.[dev,lint,docs,…]`) after prerequisites; run once per environment.
- `pytest tests`: executes the Python suites; use `-k` to target a module (e.g., `pytest tests -k data_handler`).
- `make black` / `make pylint` / `make flake8` / `make mypy`: enforce formatting, linting, and type checks aligned with CI.
- `make docs-gen`: builds the Sphinx docs into `public/` for local inspection.

## Coding Style & Naming Conventions
Follow standard Python 3.8+ practices with four-space indents and explicit imports. `black -l 120` governs formatting; avoid manual deviations. Prefer snake_case for functions and variables, PascalCase for classes, and keep module-level constants uppercase. Add targeted docstrings or comments only when the intent is non-obvious, and keep public APIs type-annotated where feasible.

## Testing Guidelines
Author tests with `pytest`, colocated under `tests/` using descriptive filenames (`test_<feature>.py`). Parametrize cases instead of duplicating fixtures, and clean up any temporary data under `mlruns/`. Run `pytest --maxfail=1 --disable-warnings` before submitting; attach regression notebooks to `examples/` only if they are executable end-to-end.

## Commit & Pull Request Guidelines
Commit messages should stay under 72 characters, written in the imperative mood (e.g., “Add task update daily data”). Reference related issues in the body and mention notable side effects. Pull requests must describe scope, validation steps, and migration considerations; include screenshots or metrics when UI or performance changes are involved. Ensure CI lint and test commands above are green before requesting review.

## Documentation & Notebooks
Keep tutorial notebooks lightweight, with reproducible cells that respect the default data paths configured in `qlib/`. When updating docs, regenerate via `make docs-gen` and check for warnings. Large datasets or credentials should be referenced through configuration examples rather than embedded in the repo.
