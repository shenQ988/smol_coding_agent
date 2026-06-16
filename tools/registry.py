"""Tool registry — merges built-in tools with MCP tools."""

from tools.filesystem import list_files, read_file, write_file, patch_file
from tools.shell import run_shell
from tools.search import search

# Risky tools that need approval before execution
RISKY_TOOLS = {"run_shell", "write_file", "patch_file"}



# Safe tools that run without asking
SAFE_TOOLS = {"list_files", "read_file", "search"}


def get_builtin_tools() -> list:
    """Return all built-in tool instances."""
    return [list_files, read_file, write_file, patch_file, run_shell, search]


def is_risky(tool_name: str) -> bool:
    """Check if a tool needs user approval."""
    return tool_name in RISKY_TOOLS