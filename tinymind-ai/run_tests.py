"""Run every test, and refuse to pass if any test was skipped.

    python run_tests.py
    python run_tests.py -v
    python run_tests.py tests.test_model

torch lives in the project virtualenv, so when the current interpreter cannot
import it this script re-executes itself under .venv/bin/python (if present)
rather than letting the network tests skip. A run with skipped tests exits
with status 1, so a partial suite cannot look like a pass.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_ROOT = os.path.join(ROOT, ".venv")
VENV_PYTHON = os.path.join(VENV_ROOT, "bin", "python")


def torch_missing():
    try:
        import torch  # noqa: F401
    except ImportError:
        return True
    return False


def re_exec_in_venv():
    """Switch to the project interpreter when torch is not importable here."""
    if not torch_missing():
        return

    if not os.path.exists(VENV_PYTHON):
        raise SystemExit(
            "torch is required for the model tests and %s does not exist.\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
            % VENV_PYTHON
        )

    # Compare prefixes, not resolved executables: the venv's python is a
    # symlink to /usr/bin/python3.12, so realpath() cannot tell them apart.
    if os.path.realpath(sys.prefix) == os.path.realpath(VENV_ROOT):
        raise SystemExit(
            "torch is missing even under %s.\n"
            "Install dependencies with: .venv/bin/pip install -r requirements.txt"
            % VENV_PYTHON
        )

    print("torch is not installed here; running the suite under %s" % VENV_PYTHON)
    os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    re_exec_in_venv()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    names = [value for value in argv if not value.startswith("-")]
    start_dir = names[0] if names else "tests"

    suite = unittest.TestLoader().discover(start_dir)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    if result.skipped:
        print(
            "\n%d test(s) were skipped; this suite must run all of them."
            % len(result.skipped)
        )
        return 1

    if not result.wasSuccessful():
        return 1

    print("\nAll %d tests ran, none skipped." % result.testsRun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
