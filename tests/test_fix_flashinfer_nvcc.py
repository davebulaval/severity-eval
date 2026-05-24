"""pytest wrapper for tests/test_fix_flashinfer_nvcc.sh.

The sed-patch logic in experiments/fix_flashinfer_nvcc.sh is too coupled
to the shell environment (mktemp, grep -rl, sed -i, while/IFS) to be
re-implemented in pure Python without losing fidelity. Instead, we ship
a Bash test that runs the same logic in a sandbox and wrap it here so
pytest picks it up.

A non-zero exit from the Bash test fails the pytest run with the stderr
from the script attached to the assertion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHELL_TEST = REPO_ROOT / "tests" / "test_fix_flashinfer_nvcc.sh"


def test_shell_sed_patch_is_correct_and_idempotent():
    """Run the bash sandbox test that validates the sed-patch logic."""
    assert SHELL_TEST.is_file(), f"missing shell test: {SHELL_TEST}"
    result = subprocess.run(
        ["bash", str(SHELL_TEST)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"shell test failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # The shell test ends with a deterministic "OK  patched 2 files..." line
    assert "OK" in result.stdout, f"unexpected output:\n{result.stdout}"


def test_shell_test_is_executable():
    """The shell test should be marked executable so users can run it directly."""
    import os
    import stat

    mode = SHELL_TEST.stat().st_mode
    assert mode & stat.S_IXUSR, "tests/test_fix_flashinfer_nvcc.sh is not executable"
    # bash is available
    assert os.access("/bin/bash", os.X_OK) or os.access("/usr/bin/bash", os.X_OK), (
        "bash not found at standard paths"
    )
