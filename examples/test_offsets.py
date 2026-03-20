"""
Test script for the new freq_offsets feature in MBFVAR.

This script tests:
1. Backward compatibility (no offsets specified - should work as before)
2. Simple offset configuration (higher frequencies start later)
3. Validation and error handling

Run with: python test_offsets.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import MBFVAR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MBFVAR


def generate_synthetic_data_with_offsets():
    """
    Generate synthetic multi-frequency data with different starting points.

    Returns:
        data_aligned: All frequencies start at the same time (for backward compatibility test)
        data_offset: Higher frequencies start later (for offset test)
        trans: Transformation specifications
        frequencies: Frequency identifiers
    """

    np.random.seed(42)

    # Generate 80 quarters of data (20 years)
    n_quarters = 80
    quarters = pd.date_range(start='2005-01-01', periods=n_quarters, freq='QE')  # Use 'QE' for pandas 2.x

    # Quarterly data: 2 variables
    q_var1 = np.cumsum(np.random.randn(n_quarters) * 0.5 + 0.1) + 10
    q_var2 = np.cumsum(np.random.randn(n_quarters) * 0.3 + 0.05) + 5

    data_q = pd.DataFrame({
        'q_gdp': q_var1,
        'q_inflation': q_var2
    }, index=quarters)

    # Monthly data: starts 20 quarters (60 months) later
    # So we have 60 quarters total, minus 20 offset = 40 quarters = 120 months
    n_months_full = n_quarters * 3  # 240 months
    n_months_offset = 120  # Start 60 months later (20 quarters)
    months = pd.date_range(start='2010-01-01', periods=n_months_offset, freq='ME')  # Use 'ME' for pandas 2.x

    m_var1 = np.cumsum(np.random.randn(n_months_offset) * 0.2 + 0.03) + 8
    m_var2 = np.cumsum(np.random.randn(n_months_offset) * 0.15 + 0.02) + 4
    m_var3 = np.cumsum(np.random.randn(n_months_offset) * 0.1 + 0.01) + 3

    data_m_offset = pd.DataFrame({
        'm_employment': m_var1,
        'm_sales': m_var2,
        'm_orders': m_var3
    }, index=months)

    # For aligned version, create monthly data from the beginning
    months_aligned = pd.date_range(start='2005-01-01', periods=n_months_full, freq='ME')  # Use 'ME'
    m_var1_aligned = np.cumsum(np.random.randn(n_months_full) * 0.2 + 0.03) + 8
    m_var2_aligned = np.cumsum(np.random.randn(n_months_full) * 0.15 + 0.02) + 4
    m_var3_aligned = np.cumsum(np.random.randn(n_months_full) * 0.1 + 0.01) + 3

    data_m_aligned = pd.DataFrame({
        'm_employment': m_var1_aligned,
        'm_sales': m_var2_aligned,
        'm_orders': m_var3_aligned
    }, index=months_aligned)

    # Weekly data: starts 32 quarters later
    # 32 quarters from base = 32*12 = 384 weeks in highest frequency
    # We need at least 384 weeks of data
    # Let's provide 400 weeks to be safe
    n_weeks_offset = 400
    weeks = pd.date_range(start='2013-01-01', periods=n_weeks_offset, freq='W')

    w_var1 = np.cumsum(np.random.randn(n_weeks_offset) * 0.1 + 0.01) + 6
    w_var2 = np.cumsum(np.random.randn(n_weeks_offset) * 0.08 + 0.005) + 2
    w_var3 = np.cumsum(np.random.randn(n_weeks_offset) * 0.05 + 0.003) + 1

    data_w_offset = pd.DataFrame({
        'w_orders': w_var1,
        'w_shipments': w_var2,
        'w_inventory': w_var3
    }, index=weeks)

    # For aligned version, create weekly data from the beginning
    n_weeks_full = n_quarters * 12  # Approximate: 12 weeks per quarter
    weeks_aligned = pd.date_range(start='2005-01-01', periods=n_weeks_full, freq='W')
    w_var1_aligned = np.cumsum(np.random.randn(n_weeks_full) * 0.1 + 0.01) + 6
    w_var2_aligned = np.cumsum(np.random.randn(n_weeks_full) * 0.08 + 0.005) + 2
    w_var3_aligned = np.cumsum(np.random.randn(n_weeks_full) * 0.05 + 0.003) + 1

    data_w_aligned = pd.DataFrame({
        'w_orders': w_var1_aligned,
        'w_shipments': w_var2_aligned,
        'w_inventory': w_var3_aligned
    }, index=weeks_aligned)

    # Transformations: all divide by 100
    trans = [
        np.array([1, 1]),         # Quarterly
        np.array([1, 1, 1]),      # Monthly
        np.array([1, 1, 1])       # Weekly
    ]

    frequencies = ["Q", "M", "W"]

    data_aligned = [data_q, data_m_aligned, data_w_aligned]
    data_offset = [data_q, data_m_offset, data_w_offset]

    return data_aligned, data_offset, trans, frequencies


def test_backward_compatibility():
    """Test that the package still works without specifying offsets."""
    print("="*70)
    print("TEST 1: Backward Compatibility (No Offsets)")
    print("="*70)

    try:
        data_aligned, _, trans, frequencies = generate_synthetic_data_with_offsets()

        # Create data object WITHOUT specifying freq_offsets
        # This should work exactly as before
        print("Creating mbfvar_data without freq_offsets parameter...")
        data_in = MBFVAR.mbfvar_data(data_aligned, trans, frequencies)

        # Check that default offsets are all zeros
        assert hasattr(data_in, 'freq_offsets'), "Should have freq_offsets attribute"
        assert data_in.freq_offsets == [0, 0, 0], f"Default offsets should be all zeros, got {data_in.freq_offsets}"

        print("  ✓ Data object created successfully")
        print(f"  ✓ Default offsets: {data_in.freq_offsets}")

        # Try to fit a small model
        print("\nFitting model (small sample for speed)...")
        model = MBFVAR.MixedFrequencyBVAR(nsim=50, nburn_perc=0.5, nlags=[6, 4], thining=1)
        hyp = [[0.1, 4.0, 1, 2.5, 4.0], [0.1, 4.0, 1, 2.5, 4.0]]
        model.fit(data_in, hyp=hyp)

        print("  ✓ Model fitted successfully")
        print("\n✅ BACKWARD COMPATIBILITY TEST PASSED!\n")
        return True

    except Exception as e:
        print(f"\n❌ BACKWARD COMPATIBILITY TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_offsets():
    """Test with explicit offsets (higher frequencies start later)."""
    print("="*70)
    print("TEST 2: Data with Offsets (Higher Frequencies Start Later)")
    print("="*70)

    try:
        _, data_offset, trans, frequencies = generate_synthetic_data_with_offsets()

        # Specify offsets: M starts 20 quarters later, W starts 32 quarters later
        freq_offsets = [0, 20, 32]

        print(f"\nData shapes:")
        print(f"  Quarterly: {data_offset[0].shape} (from 2005-01-01)")
        print(f"  Monthly:   {data_offset[1].shape} (from 2010-01-01 - 20 quarters later)")
        print(f"  Weekly:    {data_offset[2].shape} (from 2013-01-01 - 32 quarters later)")

        print(f"\nSpecified offsets: {freq_offsets}")
        print("  (M starts 20 quarters later, W starts 32 quarters later)")

        # Create data object WITH offsets
        print("\nCreating mbfvar_data with freq_offsets...")
        data_in = MBFVAR.mbfvar_data(
            data=data_offset,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=freq_offsets
        )

        print("  ✓ Data object created successfully")
        print(f"  ✓ Offsets stored: {data_in.freq_offsets}")
        print(f"  ✓ Offsets in highest freq: {data_in.freq_offsets_hf}")

        # Try to fit a small model
        print("\nFitting model with offset data...")
        model = MBFVAR.MixedFrequencyBVAR(nsim=50, nburn_perc=0.5, nlags=[6, 4], thining=1)
        hyp = [[0.1, 4.0, 1, 2.5, 4.0], [0.1, 4.0, 1, 2.5, 4.0]]
        model.fit(data_in, hyp=hyp)

        print("  ✓ Model fitted successfully with offset data")
        print("\n✅ OFFSET TEST PASSED!\n")
        return True

    except Exception as e:
        print(f"\n❌ OFFSET TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test validation and error handling."""
    print("="*70)
    print("TEST 3: Validation and Error Handling")
    print("="*70)

    passed_tests = 0
    total_tests = 5

    data_aligned, _, trans, frequencies = generate_synthetic_data_with_offsets()

    # Test 1: Wrong length
    print("\n1. Testing wrong offset length...")
    try:
        data_in = MBFVAR.mbfvar_data(
            data=data_aligned,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=[0, 10]  # Wrong length!
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
        passed_tests += 1

    # Test 2: First element not zero
    print("\n2. Testing first element not zero...")
    try:
        data_in = MBFVAR.mbfvar_data(
            data=data_aligned,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=[5, 10, 15]  # First should be 0!
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
        passed_tests += 1

    # Test 3: Negative offset
    print("\n3. Testing negative offset...")
    try:
        data_in = MBFVAR.mbfvar_data(
            data=data_aligned,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=[0, -5, 10]  # Negative not allowed!
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
        passed_tests += 1

    # Test 4: Non-integer offset
    print("\n4. Testing non-integer offset...")
    try:
        data_in = MBFVAR.mbfvar_data(
            data=data_aligned,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=[0, 5.5, 10]  # Should be integer!
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
        passed_tests += 1

    # Test 5: Offset exceeds data
    print("\n5. Testing offset exceeds available data...")
    try:
        data_in = MBFVAR.mbfvar_data(
            data=data_aligned,
            trans=trans,
            frequencies=frequencies,
            freq_offsets=[0, 200, 300]  # Way too large!
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
        passed_tests += 1

    print(f"\n{'='*70}")
    if passed_tests == total_tests:
        print(f"✅ ALL VALIDATION TESTS PASSED ({passed_tests}/{total_tests})\n")
        return True
    else:
        print(f"❌ SOME VALIDATION TESTS FAILED ({passed_tests}/{total_tests})\n")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MBFVAR OFFSET FEATURE TESTS")
    print("="*70 + "\n")

    results = []

    # Test 1: Backward compatibility
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Test 2: With offsets
    results.append(("Offset Functionality", test_with_offsets()))

    # Test 3: Validation
    results.append(("Validation & Error Handling", test_validation()))

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print("="*70)
    print(f"Total: {passed}/{total} test groups passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The offset feature is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
