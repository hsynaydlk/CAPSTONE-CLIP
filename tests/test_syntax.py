"""Check the repository entry points."""

import py_compile
from pathlib import Path


def test_workflow_files_compile():
    """Compile each workflow script."""
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "scripts").glob("*.py"))
    assert {path.name for path in scripts} == {
        "run_capstone_benchmark.py",
        "run_protocol_baselines.py",
    }
    for path in scripts:
        py_compile.compile(str(path), doraise=True)
