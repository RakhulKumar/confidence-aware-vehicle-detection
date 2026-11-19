#!/usr/bin/env python3
"""Dependency-free test runner.

`pytest tests/ -v` is the normal way to run the suite. This runner exists so
the suite can also be run where pytest is not installed:

    python tests/run_tests.py            # everything
    python tests/run_tests.py conformal  # only test files whose name contains "conformal"

It discovers every `test_*` function in tests/test_*.py, runs each one, and
exits non-zero if any fail. A minimal stand-in for `pytest.approx` and
`pytest.raises` is injected when pytest itself is missing.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def _install_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            tol = max(self.abs, self.rel * abs(self.expected))
            return abs(other - self.expected) <= tol

        def __repr__(self):
            return f"approx({self.expected})"

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__} to be raised")
            if not issubclass(exc_type, self.exc):
                return False
            if self.match is not None:
                import re
                if not re.search(self.match, str(exc)):
                    raise AssertionError(f"{exc!r} does not match {self.match!r}")
            return True

    shim = types.ModuleType("pytest")
    shim.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)
    shim.raises = lambda exc, match=None: _Raises(exc, match)
    sys.modules["pytest"] = shim


def main() -> int:
    _install_pytest_shim()
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in HERE.glob("test_*.py") if pattern in p.name)
    passed, failed = 0, []
    for path in files:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            failed.append((path.name, "<import>", traceback.format_exc()))
            continue
        for name, fn in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            kwargs = {}
            tmp = None
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp.name)
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS  {path.name}::{name}")
            except Exception:
                failed.append((path.name, name, traceback.format_exc()))
                print(f"FAIL  {path.name}::{name}")
            finally:
                if tmp is not None:
                    tmp.cleanup()
    print(f"\n{passed} passed, {len(failed)} failed")
    for fname, name, tb in failed:
        print(f"\n--- {fname}::{name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
