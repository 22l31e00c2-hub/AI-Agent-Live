"""Guarded execution layer.

Generating a script is always safe, so the agent always does it. Running
something is opt-in, restricted to an allowlist of command-line tools, and
never goes through a shell.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional

from .types import ExecuteSpec

# Only these binaries may ever be started by the agent.
ALLOWED_TOOLS = {"ogr2ogr", "gdalinfo", "qgis_process"}


@dataclass
class Outcome:
    ran: bool
    reason: str = ""
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def capabilities() -> Dict[str, object]:
    """What this machine can actually do, right now."""
    return {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "ogr2ogr": shutil.which("ogr2ogr"),
        "qgis_process": shutil.which("qgis_process"),
        "pyautocad": _has_module("pyautocad") or _has_module("comtypes"),
        "pyrevit": shutil.which("pyrevit"),
        "pillow": _has_module("PIL"),
        "tesseract": shutil.which("tesseract"),
    }


def run_spec(spec: Optional[ExecuteSpec], timeout: int = 600) -> Outcome:
    if spec is None:
        return Outcome(ran=False, reason="this recipe has no executable step")

    tool = spec.argv[0]
    if tool not in ALLOWED_TOOLS:
        return Outcome(
            ran=False,
            reason="refused: %r is not on the runner allowlist" % tool,
        )

    resolved = shutil.which(tool)
    if not resolved:
        return Outcome(
            ran=False,
            reason="%s is not installed on this machine" % tool,
        )

    argv = [resolved] + spec.argv[1:]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome(ran=False, reason="command timed out after %ds" % timeout)
    except OSError as exc:
        return Outcome(ran=False, reason="could not start %s: %s" % (tool, exc))

    return Outcome(
        ran=True,
        returncode=completed.returncode,
        stdout=(completed.stdout or "").strip(),
        stderr=(completed.stderr or "").strip(),
    )
