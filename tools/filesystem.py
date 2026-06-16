"""File system tools — read, write, list, patch."""

from pathlib import Path
from langchain_core.tools import tool

# Will be set at startup
WORKSPACE_ROOT: Path = Path.cwd()


def set_workspace_root(path: Path):
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = path


def _safe_path(path_str: str) -> Path:
    """Ensure path stays inside workspace."""
    resolved = (WORKSPACE_ROOT / path_str).resolve()
    if not resolved.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return resolved


@tool
def list_files(path: str = ".") -> str:
    """List files and directories in the given path.

    Args:
        path: Relative path from workspace root. Defaults to current directory.
    """
    target = _safe_path(path)
    if not target.exists():
        return f"Error: {path} does not exist"

    entries = sorted(target.iterdir())
    lines = []
    for entry in entries[:100]:  # cap at 100 entries
        prefix = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{prefix} {entry.relative_to(WORKSPACE_ROOT)}")
    return "\n".join(lines) or "(empty directory)"


@tool
def read_file(path: str, start: int = 1, end: int = 200) -> str:
    """Read lines from a text file.

    Args:
        path: Relative path to the file.
        start: First line to read (1-indexed). Default 1.
        end: Last line to read (inclusive). Default 200.
    """
    target = _safe_path(path)
    if not target.is_file():
        return f"Error: {path} is not a file"

    try:
        lines = target.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    start = max(1, start)
    end = min(end, len(lines))
    selected = lines[start - 1:end]

    # Add line numbers (like an IDE)
    numbered = [f"{i:4d} | {line}" for i, line in enumerate(selected, start)]
    header = f"File: {path} (lines {start}-{end} of {len(lines)})"
    return header + "\n" + "\n".join(numbered)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: Relative path where the file will be written.
        content: The full content to write.
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


@tool
def patch_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact occurrence of old_text with new_text in a file.

    Args:
        path: Relative path to the file to patch.
        old_text: The exact text to find (must appear exactly once).
        new_text: The replacement text.
    """
    target = _safe_path(path)
    if not target.is_file():
        return f"Error: {path} does not exist"

    content = target.read_text()
    count = content.count(old_text)

    if count == 0:
        return f"Error: old_text not found in {path}"
    if count > 1:
        return f"Error: old_text appears {count} times in {path} (must be exactly 1)"

    new_content = content.replace(old_text, new_text, 1)
    target.write_text(new_content)
    return f"Patched {path}: replaced 1 occurrence"