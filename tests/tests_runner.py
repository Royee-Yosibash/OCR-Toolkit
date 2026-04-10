import sys
import unittest

from consts import APP_ROOT


def run_all_tests():
    """
    Discover and run all test files named 'test*.py'
    starting from the project root.
    """

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=APP_ROOT,
        pattern="test_*.py"
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit code for CI systems (GitHub, etc.)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_all_tests()
