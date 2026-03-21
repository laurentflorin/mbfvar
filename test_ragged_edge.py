"""
Test ragged-edge handling in MBFVAR
"""
import sys
import numpy as np

sys.path.insert(0, '/home/runner/work/MBFVAR/MBFVAR')

from MBFVAR.mfbvar_funcs import _filter_valid_var_rows, calc_yyact, mdd_


def test_filter_valid_var_rows():
    """Test the _filter_valid_var_rows function."""
    print("Testing _filter_valid_var_rows...")

    # Create test data with some NaN values in the last rows
    nobs = 10
    nv = 2
    nlags = 2

    YYact = np.random.randn(nobs, nv)
    XXact = np.random.randn(nobs, nv * nlags + 1)

    # Add NaN values to simulate ragged edge
    YYact[-2:, 0] = np.nan  # Last 2 rows have NaN in first variable
    XXact[-1, :] = np.nan   # Last row has NaN in all regressors

    print(f"  Original YYact shape: {YYact.shape}")
    print(f"  Original XXact shape: {XXact.shape}")
    print(f"  YYact has NaN: {np.isnan(YYact).any()}")
    print(f"  XXact has NaN: {np.isnan(XXact).any()}")

    # Filter out invalid rows
    YYact_f, XXact_f, valid = _filter_valid_var_rows(YYact, XXact)

    print(f"  Filtered YYact shape: {YYact_f.shape}")
    print(f"  Filtered XXact shape: {XXact_f.shape}")
    print(f"  Number of valid rows: {np.sum(valid)}")
    print(f"  Filtered YYact has NaN: {np.isnan(YYact_f).any()}")
    print(f"  Filtered XXact has NaN: {np.isnan(XXact_f).any()}")

    # Verify no NaN values remain
    assert not np.isnan(YYact_f).any(), "Filtered YYact should not contain NaN"
    assert not np.isnan(XXact_f).any(), "Filtered XXact should not contain NaN"
    assert YYact_f.shape[0] == XXact_f.shape[0], "Filtered arrays should have same number of rows"
    assert YYact_f.shape[0] < YYact.shape[0], "Filtered array should have fewer rows"

    print("  ✓ _filter_valid_var_rows test passed!")
    return True


def test_calc_yyact_with_ragged_edge():
    """Test calc_yyact with ragged-edge data."""
    print("\nTesting calc_yyact with ragged edge...")

    # Create simple test data
    nv = 3
    nlags = 2
    T0 = 20
    nobs = 30

    # Create YY matrix with NaN values at the end (ragged edge)
    YY = np.random.randn(T0 + nobs, nv)
    YY[-5:, 0] = np.nan  # Last 5 rows have NaN in first variable
    YY[-2:, 1] = np.nan  # Last 2 rows have NaN in second variable

    print(f"  YY shape: {YY.shape}")
    print(f"  YY has NaN in last rows: {np.isnan(YY[-10:, :]).any()}")

    # Create spec and hyperparameters
    spec = np.array([nlags, T0, 1, nv, nobs])
    hyp = np.array([0.1, 2.0, 1, 1.0, 1.0])

    try:
        YYact, YYdum, XXact, XXdum = calc_yyact(hyp, YY, spec)

        print(f"  YYact shape: {YYact.shape}")
        print(f"  XXact shape: {XXact.shape}")
        print(f"  YYact has NaN: {np.isnan(YYact).any()}")
        print(f"  XXact has NaN: {np.isnan(XXact).any()}")

        # Verify no NaN values in output
        assert not np.isnan(YYact).any(), "YYact should not contain NaN"
        assert not np.isnan(XXact).any(), "XXact should not contain NaN"
        assert YYact.shape[0] > 0, "YYact should have at least some rows"

        print("  ✓ calc_yyact with ragged edge test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Test failed with error: {e}")
        raise


def test_mdd_with_ragged_edge():
    """Test mdd_ with ragged-edge data."""
    print("\nTesting mdd_ with ragged edge...")

    # Create simple test data
    nv = 3
    nlags = 2
    T0 = 20
    nobs = 30

    # Create YY matrix with NaN values at the end (ragged edge)
    YY = np.random.randn(T0 + nobs, nv)
    YY[-5:, 0] = np.nan  # Last 5 rows have NaN in first variable
    YY[-2:, 1] = np.nan  # Last 2 rows have NaN in second variable

    print(f"  YY shape: {YY.shape}")
    print(f"  YY has NaN in last rows: {np.isnan(YY[-10:, :]).any()}")

    # Create spec and hyperparameters
    spec = np.array([nlags, T0, 1, nv, nobs])
    hyp = np.array([0.1, 2.0, 1, 1.0, 1.0])

    try:
        mdd, YYact, YYdum, XXact, XXdum = mdd_(hyp, YY, spec)

        print(f"  MDD value: {mdd}")
        print(f"  YYact shape: {YYact.shape}")
        print(f"  XXact shape: {XXact.shape}")
        print(f"  YYact has NaN: {np.isnan(YYact).any()}")
        print(f"  XXact has NaN: {np.isnan(XXact).any()}")

        # Verify no NaN values in output
        assert not np.isnan(YYact).any(), "YYact should not contain NaN"
        assert not np.isnan(XXact).any(), "XXact should not contain NaN"
        assert not np.isnan(mdd), "MDD should not be NaN"
        assert not np.isinf(mdd), "MDD should not be inf"
        assert YYact.shape[0] > 0, "YYact should have at least some rows"

        print("  ✓ mdd_ with ragged edge test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Test failed with error: {e}")
        raise


def run_all_tests():
    """Run all ragged-edge tests."""
    print("="*70)
    print("RAGGED-EDGE HANDLING TESTS")
    print("="*70)

    try:
        test_filter_valid_var_rows()
        test_calc_yyact_with_ragged_edge()
        test_mdd_with_ragged_edge()

        print("\n" + "="*70)
        print("✓ ALL RAGGED-EDGE TESTS PASSED!")
        print("="*70)
        return True

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ TESTS FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
