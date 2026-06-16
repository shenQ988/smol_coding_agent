"""Search tool — grep/ripgrep over the codebase."""

import shutil
import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKSPACE_ROOT: Path = Path.cwd()


def set_workspace_root(path: Path):
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = path


@tool
def search(pattern: str, path: str = ".") -> str:
    """Search for a pattern in files using ripgrep (rg) or grep fallback.

    Args:
        pattern: The regex pattern to search for.
        path: Directory to search in. Defaults to workspace root.
    """
    target = (WORKSPACE_ROOT / path).resolve()

    if shutil.which("rg"):
        cmd = ["rg", "--no-heading", "-n", "--max-count=50", pattern, str(target)]
    else:
        cmd = ["grep", "-rn", "--max-count=50", pattern, str(target)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        if len(output) > 4000:
            output = output[:4000] + "\n... (truncated)"
        return output or "(no matches)"
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error: {e}"