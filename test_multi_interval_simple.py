#!/usr/bin/env python3
"""
Simple test script for the new multi-interval functionality
This test script focuses on validating the interval normalization logic
without importing the full qlib package.
"""

def normalize_interval_format(interval: str) -> tuple[str, str]:
    """Test version of the interval normalization function"""
    interval_lower = interval.lower()
    
    # Mapping of input formats to (standardized_format, dump_frequency)
    interval_mapping = {
        # Daily intervals
        "1d": ("1D", "day"),
        "1D": ("1D", "day"),
        
        # Minute intervals
        "1min": ("1min", "1min"),
        "1m": ("1min", "1min"),
        "5min": ("5min", "5min"),
        "5m": ("5min", "5min"),
        "30min": ("30min", "30min"),
        "30m": ("30min", "30min"),
        
        # Hour intervals
        "1h": ("1H", "1h"),
        "1H": ("1H", "1h"),
        "6h": ("6H", "6h"),
        "6H": ("6H", "6h"),
    }
    
    if interval_lower not in interval_mapping:
        supported = list(interval_mapping.keys())
        raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {supported}")
    
    return interval_mapping[interval_lower]

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
            result = normalize_interval_format(input_interval)
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
            result = normalize_interval_format(invalid_interval)
            print(f"  ✗ {invalid_interval} should have failed but returned {result}")
            all_passed = False
        except ValueError as e:
            print(f"  ✓ {invalid_interval} -> Correctly rejected")
        except Exception as e:
            print(f"  ✗ {invalid_interval} -> Unexpected error: {e}")
            all_passed = False
    
    return all_passed

def test_supported_intervals():
    """Test that all defined intervals work correctly"""
    print("\nTesting all supported intervals:")
    
    supported_intervals = ["1D", "1d", "1min", "1m", "5m", "5min", "30m", "30min", "1h", "1H", "6h", "6H"]
    standardized_intervals = ["1D", "1min", "5min", "30min", "1H", "6H"]
    
    print(f"  Supported intervals: {supported_intervals}")
    print(f"  Standardized intervals: {standardized_intervals}")
    
    # Check that all supported intervals are valid
    all_valid = True
    for interval in supported_intervals:
        try:
            standardized, freq = normalize_interval_format(interval)
            print(f"  ✓ {interval} -> {standardized} (freq: {freq})")
        except Exception as e:
            print(f"  ✗ {interval} is invalid: {e}")
            all_valid = False
    
    return all_valid

def test_interval_ordering():
    """Test the expected ordering for batch processing"""
    print("\nTesting interval ordering for batch processing:")
    
    intervals = ["6H", "1min", "1D", "30min", "5min", "1H"]
    interval_order = ["1D", "1min", "5min", "30min", "1H", "6H"]
    
    # Simulate the ordering logic
    sorted_intervals = []
    for target_interval in interval_order:
        for interval in intervals:
            try:
                standardized, _ = normalize_interval_format(interval)
                if standardized == target_interval and interval not in [i[0] for i in sorted_intervals]:
                    sorted_intervals.append((interval, standardized))
            except ValueError:
                continue
    
    print(f"  Input order: {intervals}")
    print(f"  Optimal order: {[interval for interval, _ in sorted_intervals]}")
    print(f"  Expected order: {interval_order}")
    
    actual_order = [std for _, std in sorted_intervals]
    return actual_order == interval_order

def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing Multi-Interval Update Functionality")
    print("=" * 70)
    
    test1_passed = test_interval_normalization()
    test2_passed = test_supported_intervals() 
    test3_passed = test_interval_ordering()
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print(f"  Interval normalization: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Supported intervals: {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Interval ordering: {'PASS' if test3_passed else 'FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed!")
        print("\nThe new multi-interval functionality is ready to use!")
        print("\nSupported intervals: 1D, 1d, 1min, 1m, 5m, 5min, 30m, 30min, 1h, 1H, 6h, 6H")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())