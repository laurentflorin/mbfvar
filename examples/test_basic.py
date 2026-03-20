"""
Simple tests for MUFBVAR package

These tests verify basic functionality of the package.
Run with: python test_basic.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import MUFBVAR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MUFBVAR


def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")

    assert hasattr(MUFBVAR, 'mufbvar_data'), "Should have mufbvar_data class"
    assert hasattr(MUFBVAR, 'multifrequency_var'), "Should have multifrequency_var class"

    print("  ✓ All imports successful")


def test_mufbvar_data_with_real_data():
    """Test mufbvar_data with the actual hist.xlsx file."""
    print("Testing mufbvar_data with real data...")

    # Check if hist.xlsx exists
    if not os.path.exists("hist.xlsx"):
        print("  ⚠ Skipping: hist.xlsx not found")
        return None

    try:
        # Load data
        io_data = "hist.xlsx"
        frequencies = ["Q", "M", "W"]

        data = []
        for freq in frequencies:
            data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
            data.append(data_temp)

        # Specify transformations
        trans = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]

        # Initialize
        data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)

        assert data_in is not None, "Data object should be created"
        assert len(data_in.frequencies) == 3, "Should have 3 frequencies"

        print("  ✓ mufbvar_data with real data successful")
        return data_in

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_multifrequency_var_initialization():
    """Test that multifrequency_var can be initialized."""
    print("Testing multifrequency_var initialization...")

    nsim = 100
    nburn = 0.5
    nlags = [6, 4]
    thining = 1

    model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

    # Check attributes
    assert model.nsim == nsim, "nsim should match"
    assert model.nburn_perc == nburn, "nburn_perc should match"
    assert model.nlags == nlags, "nlags should match"
    assert model.thining == thining, "thining should match"

    print("  ✓ multifrequency_var initialization successful")
    return model


def test_model_fit_with_real_data():
    """Test model fitting with real data."""
    print("Testing model fitting with real data...")

    # Load real data
    data_in = test_mufbvar_data_with_real_data()
    if data_in is None:
        print("  ⚠ Skipping: No data available")
        return None

    # Initialize model with small number of simulations
    nsim = 100  # Small for testing
    nburn = 0.5
    nlags = [6, 4]
    thining = 1

    model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

    # Hyperparameters
    hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]

    try:
        model.fit(data_in, hyp=hyp)
        print("  ✓ Model fitting with real data successful")
        return model, data_in
    except Exception as e:
        print(f"  ✗ Model fitting failed: {e}")
        raise


def test_forecast_with_real_data():
    """Test forecasting with real data."""
    print("Testing forecast generation with real data...")

    result = test_model_fit_with_real_data()
    if result is None:
        print("  ⚠ Skipping: No fitted model available")
        return

    model, data_in = result
    H = 12

    try:
        model.forecast(H)
        print("  ✓ Forecast generation successful")
    except Exception as e:
        print(f"  ✗ Forecast failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING MUFBVAR BASIC TESTS")
    print("="*70 + "\n")

    tests = [
        ("Import Tests", test_imports),
        ("Model Initialization", test_multifrequency_var_initialization),
        ("Data with Real File", test_mufbvar_data_with_real_data),
        ("Model Fitting with Real Data", test_model_fit_with_real_data),
        ("Forecasting with Real Data", test_forecast_with_real_data),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            result = test_func()
            if result is None and "Real" in test_name:
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print(f"Skipped: {skipped}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED (or skipped)!")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        sys.exit(1)

    print("="*70 + "\n")


if __name__ == "__main__":
    # Change to examples directory if not already there
    if os.path.basename(os.getcwd()) != "examples":
        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(example_dir)

    run_all_tests()
