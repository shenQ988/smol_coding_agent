"""Shell tool — runs bash commands with timeout."""

import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKSPACE_ROOT: Path = Path.cwd()


def set_workspace_root(path: Path):
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = path


@tool
def run_shell(command: str, timeout: int = 30) -> str:
    """Run a shell command in the workspace directory.
    CAUTION: This tool can modify files and system state.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds. Default 30.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        # Truncate long output 
        if len(output) > 5000:
            output = output[:5000] + "\n... (truncated)"
        return f"exit_code: {result.returncode}\n{output}" or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"