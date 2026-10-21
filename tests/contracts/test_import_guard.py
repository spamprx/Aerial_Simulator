"""Agent-side packages must never import ground truth (aerial.world) — PLAN §6."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "aerial"
GUARDED = ["agents", "epistemic", "mapping", "planning"]


@pytest.mark.parametrize("pkg", GUARDED)
def test_no_world_imports(pkg):
    for f in (ROOT / pkg).rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                assert not (mod.startswith("aerial.world") or mod.lstrip(".").startswith("world")), f"{f} imports {mod}"
