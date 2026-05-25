"""Static tests for experiments/cleanup.sh.

The --kill section spawns nothing testable in CI, but we can still
guard against the specific regression that recurred multiple times in
this repo: the kill pattern silently dropping a vLLM/wandb subprocess
class, leaving GPU+VRAM allocations stranded after a "successful"
cleanup. These tests grep the script's content so that any future
edit that removes EngineCore or wandb-core from the pattern fails CI
before it ships.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEANUP_SH = REPO_ROOT / "experiments" / "cleanup.sh"


@pytest.fixture(scope="module")
def cleanup_source() -> str:
    assert CLEANUP_SH.is_file(), f"missing script: {CLEANUP_SH}"
    return CLEANUP_SH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def kill_pattern(cleanup_source: str) -> str:
    """Extract the regex assigned to PATTERN= inside the --kill section.

    The script defines PATTERN=... once in the DO_KILL block. We grab the
    value between the single quotes so the tests don't break when the
    surrounding code is reformatted.
    """
    m = re.search(r"^\s*PATTERN='([^']+)'", cleanup_source, re.MULTILINE)
    assert m is not None, "PATTERN='...' assignment not found in cleanup.sh"
    return m.group(1)


@pytest.mark.parametrize(
    "subprocess_name",
    [
        # vLLM spawns one EngineCore per LLM(). Without this in the kill
        # pattern, the GPU stays allocated after the parent dies.
        "EngineCore",
        # wandb's local agent. Survives parent SIGTERM and keeps writing
        # to ./wandb/, blocking the next run with stale state.
        "wandb-core",
        # The two evaluator entry points.
        "evaluate_models",
        "run_local_smoke",
    ],
)
def test_kill_pattern_includes_critical_subprocess(
    kill_pattern: str, subprocess_name: str
):
    """Each name must literally appear inside the kill pattern.

    If this fails, a previous cleanup script will leave zombies behind.
    Reason: past sessions repeatedly required manual
    `pkill -9 -f EngineCore` / `pkill -9 -f wandb-core` after running
    `cleanup.sh --apply --kill`.
    """
    assert subprocess_name in kill_pattern, (
        f"{subprocess_name!r} not in --kill pattern. Current pattern: {kill_pattern}"
    )


def test_cleanup_script_is_executable():
    """User invokes it as ./experiments/cleanup.sh, so the +x bit matters."""
    mode = CLEANUP_SH.stat().st_mode
    assert mode & stat.S_IXUSR, "experiments/cleanup.sh is not executable"


def test_cleanup_dry_run_is_default(cleanup_source: str):
    """Without --apply the script must be read-only.

    Regression guard: a past iteration set APPLY=true by default and
    deleted batch_error JSONs on every invocation. The dry-run default
    is the safety net.
    """
    assert re.search(r"^\s*APPLY=false\b", cleanup_source, re.MULTILINE), (
        "APPLY must default to false (dry-run) in cleanup.sh"
    )


def test_purge_hf_requires_apply(cleanup_source: str):
    """--purge-hf is destructive (~150 GB). It must be gated by --apply.

    The relevant condition is `if [[ "$DO_PURGE_HF" == "true" && "$APPLY"
    == "true" ]]`. We assert that the rm -rf line is only reachable when
    both flags are set.
    """
    # Find the block whose body runs `rm -rf "$HF_HOME_DIR/hub"` and confirm
    # the controlling `if` line tests both DO_PURGE_HF and APPLY = true.
    rm_match = re.search(
        r'(if\s+\[\[\s+"\$DO_PURGE_HF"[^\n]*\n(?:[^\n]*\n){0,5}\s*rm -rf\s+"\$HF_HOME_DIR/hub")',
        cleanup_source,
    )
    assert rm_match is not None, "HF cache rm -rf block not found"
    if_line = rm_match.group(1).splitlines()[0]
    assert "DO_PURGE_HF" in if_line and 'APPLY" == "true"' in if_line, (
        f"HF cache rm -rf is not gated by both DO_PURGE_HF and APPLY=true. "
        f"Gate line: {if_line!r}"
    )
