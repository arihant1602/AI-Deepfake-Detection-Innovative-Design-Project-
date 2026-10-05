"""
Unified Test Runner for AI Deepfake Detection Project
=====================================================
Discovers and executes all unit and integration test suites:
  - Layer 1: test_host_integrity.py (host & camera attestation)
  - Layer 2: test_sensor_noise_profiler.py (PRNU Analysis)
  - Layer 3: test_temporal_consistency.py (Temporal & Frequency Analysis)
  - Layer 5: test_pipeline.py (Gated Verification Pipeline & Attestation)
"""

import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "test_host_integrity",
        "test_sensor_noise_profiler",
        "test_temporal_consistency",
        "test_pipeline",
    ]

    print("=" * 70)
    print("AI Deepfake Detection & Injection Attack Defense - Test Suite")
    print("=" * 70)

    for mod_name in test_modules:
        try:
            mod = __import__(mod_name)
            tests = loader.loadTestsFromModule(mod)
            suite.addTests(tests)
            print(f" Loaded module: {mod_name} ({tests.countTestCases()} test cases)")
        except Exception as e:
            print(f" Failed to load module {mod_name}: {e}")
            sys.exit(1)

    print("-" * 70)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 70)

    if result.wasSuccessful():
        print(f"SUCCESS: All {result.testsRun} tests passed successfully!")
        sys.exit(0)
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
