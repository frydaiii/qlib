#!/usr/bin/env python3
"""
Test script for the new multi-interval update functionality
"""

import sys
from pathlib import Path

# Add the scripts directory to Python path
scripts_dir = Path(__file__).parent / "scripts" / "data_collector" / "vnstock"
sys.path.insert(0, str(scripts_dir))

from collector import Run

def test_interval_normalization():
    """Test interval format normalization"""
    print("Testing interval normalization:")
    
    test_cases = [
        ("1D", ("1D", "day")),
        ("1d", ("1D", "day")),
        ("1min", ("1min", "1min")),
        ("1m", ("1min", "1min")),
        ("5m", ("5min", "5min")),
        ("5min", ("5min", "5min")),
        ("30m", ("30min", "30min")),
        ("30min", ("30min", "30min")),
        ("1h", ("1H", "1h")),
        ("1H", ("1H", "1h")),
        ("6h", ("6H", "6h")),
        ("6H", ("6H", "6h")),
    ]
    
    all_passed = True
    for input_interval, expected_output in test_cases:
        try:
            result = Run._normalize_interval_format(input_interval)
            if result == expected_output:
                print(f"  ✓ {input_interval} -> {result}")
            else:
                print(f"  ✗ {input_interval} -> {result} (expected {expected_output})")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {input_interval} -> Error: {e}")
            all_passed = False
    
    # Test invalid intervals
    invalid_cases = ["2D", "15min", "2h", "invalid"]
    print("\nTesting invalid intervals:")
    for invalid_interval in invalid_cases:
        try:
            result = Run._normalize_interval_format(invalid_interval)
            print(f"  ✗ {invalid_interval} should have failed but returned {result}")
            all_passed = False
        except ValueError as e:
            print(f"  ✓ {invalid_interval} -> Correctly rejected: {str(e)[:50]}...")
        except Exception as e:
            print(f"  ✗ {invalid_interval} -> Unexpected error: {e}")
            all_passed = False
    
    return all_passed

def test_supported_intervals():
    """Test supported intervals methods"""
    print("\nTesting supported intervals methods:")
    
    supported = Run.get_supported_intervals()
    standardized = Run.get_standardized_intervals()
    
    print(f"  Supported intervals: {supported}")
    print(f"  Standardized intervals: {standardized}")
    
    # Check that all standardized intervals are valid
    all_valid = True
    for interval in standardized:
        try:
            Run._normalize_interval_format(interval)
            print(f"  ✓ {interval} is valid")
        except Exception as e:
            print(f"  ✗ {interval} is invalid: {e}")
            all_valid = False
    
    return all_valid

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Multi-Interval Update Functionality")
    print("=" * 60)
    
    test1_passed = test_interval_normalization()
    test2_passed = test_supported_intervals()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Interval normalization: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Supported intervals: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())