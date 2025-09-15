# Task Completion Checklist

## Before Submitting Code

### Code Quality
- [ ] Run `make black` to format code
- [ ] Run `make lint` to check for style issues
- [ ] Run `make mypy` for type checking
- [ ] Fix any linting or type errors

### Testing
- [ ] Run relevant unit tests: `pytest tests/test_<module>.py`
- [ ] Run integration tests if applicable
- [ ] Ensure all tests pass
- [ ] Add new tests for new functionality
- [ ] Verify test coverage is maintained

### Documentation
- [ ] Update docstrings for new/modified functions
- [ ] Add examples in docstrings for CLI commands
- [ ] Update README.md if adding new features
- [ ] Update CHANGELOG.md for significant changes

### Data Collection Specific Tasks
- [ ] Test with small dataset first (use --limit_nums for debugging)
- [ ] Verify data normalization produces expected format
- [ ] Test binary dump process completes successfully
- [ ] Validate calendar generation for new time intervals
- [ ] Check compatibility with existing qlib data structures

### Final Checks
- [ ] Commit messages follow conventional commit format
- [ ] No debug print statements or temporary code
- [ ] Configuration parameters documented
- [ ] Error handling covers edge cases
- [ ] Performance considerations addressed for large datasets

## Post-Development
- [ ] Monitor data collection jobs for stability
- [ ] Validate output data quality
- [ ] Update deployment documentation if needed