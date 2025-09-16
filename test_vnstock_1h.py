#!/usr/bin/env python3
"""
Simple test script to verify VNStockCollectorVN1H implementation
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_collector.vnstock.collector import VNStockCollectorVN1H, Run

def test_vnstock_collector_1h():
    """Test VNStockCollectorVN1H class instantiation and basic functionality"""

    print("Testing VNStockCollectorVN1H implementation...")

    # Test class instantiation
    try:
        collector = VNStockCollectorVN1H(
            save_dir=Path("./test_data"),
            start="2024-01-01",
            end="2024-01-05",
            interval="1H",
            max_workers=1
        )
        print("✓ VNStockCollectorVN1H instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate VNStockCollectorVN1H: {e}")
        return False

    # Test interval property
    if collector.interval == "1H":
        print("✓ Interval property set correctly")
    else:
        print(f"✗ Interval property incorrect: {collector.interval}")
        return False

    # Test timezone property
    if collector._timezone == "Asia/Ho_Chi_Minh":
        print("✓ Timezone property set correctly")
    else:
        print(f"✗ Timezone property incorrect: {collector._timezone}")
        return False

    return True

def test_run_class_1h():
    """Test Run class with 1H interval"""

    print("\nTesting Run class with 1H interval...")

    try:
        run = Run(interval="1H", region="VN")
        print("✓ Run class instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate Run class: {e}")
        return False

    # Test collector class name
    expected_collector_name = "VNStockCollectorVN1H"
    if run.collector_class_name == expected_collector_name:
        print("✓ Collector class name generated correctly")
    else:
        print(f"✗ Collector class name incorrect: {run.collector_class_name}, expected: {expected_collector_name}")
        return False

    # Test normalize class name
    expected_normalize_name = "VNStockNormalizeVN1H"
    if run.normalize_class_name == expected_normalize_name:
        print("✓ Normalize class name generated correctly")
    else:
        print(f"✗ Normalize class name incorrect: {run.normalize_class_name}, expected: {expected_normalize_name}")
        return False

    return True

def test_interval_mapping():
    """Test interval format normalization"""

    print("\nTesting interval format normalization...")

    try:
        from scripts.data_collector.vnstock.collector import Run

        # Test 1H normalization
        normalized_1h, freq_1h = Run._normalize_interval_format("1H")
        if normalized_1h == "1H" and freq_1h == "1h":
            print("✓ 1H interval normalized correctly")
        else:
            print(f"✗ 1H interval normalization failed: {normalized_1h}, {freq_1h}")
            return False

        # Test 1h normalization
        normalized_1h_lower, freq_1h_lower = Run._normalize_interval_format("1h")
        if normalized_1h_lower == "1H" and freq_1h_lower == "1h":
            print("✓ 1h interval normalized correctly")
        else:
            print(f"✗ 1h interval normalization failed: {normalized_1h_lower}, {freq_1h_lower}")
            return False

        return True

    except Exception as e:
        print(f"✗ Interval normalization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running VNStockCollectorVN1H tests...\n")

    success = True
    success &= test_vnstock_collector_1h()
    success &= test_run_class_1h()
    success &= test_interval_mapping()

    if success:
        print("\n🎉 All tests passed! VNStockCollectorVN1H implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

    sys.exit(0 if success else 1)