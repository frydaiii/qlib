# Qlib Project Overview

## Purpose
Qlib is an AI-oriented quantitative investment platform that aims to realize the potential, empower research, and create value using AI technologies in quantitative investment. It supports diverse machine learning modeling paradigms including supervised learning, market dynamics modeling, and reinforcement learning.

## Key Features
- Full ML pipeline of data processing, model training, back-testing
- Covers the entire chain of quantitative investment: alpha seeking, risk modeling, portfolio optimization, and order execution
- Data collection from multiple sources (Yahoo Finance, VNStock for Vietnamese market, etc.)
- Support for multiple time intervals (1D daily, 1min intraday)
- Automated data normalization and binary format storage for efficient access

## Architecture
The project follows a modular architecture with:
- Data collectors for different sources/regions (scripts/data_collector/)
- Base classes for collectors, normalizers, and runners
- Data dumping utilities for converting to efficient binary formats
- Model implementations and ML pipeline components
- Testing framework with comprehensive test coverage

## Tech Stack
- Python 3.8+ 
- Core dependencies: pandas, numpy, pyyaml, mlflow, lightgbm, gym, cvxpy
- Build system: setuptools with Cython extensions
- Documentation: Sphinx with ReadTheDocs
- Development tools: pytest, black, pylint, mypy, pre-commit hooks