"""Smoke-tests the Python code blocks in README.md and docs/guide/quickstart.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from mktestdocs import grab_code_blocks

import rucola
from rucola import NormalizationConfig, RunConfig

pytestmark = pytest.mark.slow

_README = Path(__file__).parent.parent / "README.md"
_QUICKSTART = Path(__file__).parent.parent / "docs" / "guide" / "quickstart.md"

# Blocks referencing external data files unavailable in tests
_SKIP_WHEN_CONTAINS = (
    "values.csv",
    "stations.csv",
    "climate.duckdb",
    ".parquet",
    "values_df",
    "stations_df",
)


def _runnable_blocks(path: Path) -> list[str]:
    return [b for b in grab_code_blocks(path.read_text()) if not any(marker in b for marker in _SKIP_WHEN_CONTAINS)]


_readme_blocks = _runnable_blocks(_README)
_quickstart_blocks = _runnable_blocks(_QUICKSTART)


@pytest.fixture
def doc_globs(rucola_instance: rucola.Rucola, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Pre-populated namespace for executing doc code blocks."""
    monkeypatch.chdir(tmp_path)
    detection = rucola_instance.run()
    result = detection.normalize()
    return {
        "rucola": rucola,
        "pl": pl,
        "r": rucola_instance,
        "detection": detection,
        "result": result,
        "RunConfig": RunConfig,
        "NormalizationConfig": NormalizationConfig,
    }


@pytest.mark.parametrize("block", _readme_blocks, ids=[f"readme-{i}" for i in range(len(_readme_blocks))])
def test_readme_block(block: str, doc_globs: dict[str, Any]) -> None:
    """Execute one README code block in a pre-populated namespace."""
    exec(compile(block, "<readme>", "exec"), doc_globs)  # noqa: S102


@pytest.mark.parametrize("block", _quickstart_blocks, ids=[f"quickstart-{i}" for i in range(len(_quickstart_blocks))])
def test_quickstart_block(block: str, doc_globs: dict[str, Any]) -> None:
    """Execute one quickstart code block in a pre-populated namespace."""
    exec(compile(block, "<quickstart>", "exec"), doc_globs)  # noqa: S102
