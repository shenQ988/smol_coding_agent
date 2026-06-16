"""Workspace context — scans the git repo for context."""

import subprocess
from pathlib import Path


class WorkspaceContext:
    """Gathers git branch, status, recent commits, and project docs."""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.repo_root = self._git(["rev-parse", "--show-toplevel"], str(cwd))
        self.branch = self._git(["branch", "--show-current"], "-")
        self.status = self._git(["status", "--short"], "clean")[:1500]
        self.recent_commits = [
            line for line in
            self._git(["log", "--oneline", "-5"]).splitlines()
            if line
        ]
        self.project_docs = self._scan_docs()

    def _git(self, args: list[str], fallback: str = "") -> str:
        """Run a git command safely, return fallback on error."""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.cwd,
                capture_output=True, text=True, check=True, timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return fallback

    def _scan_docs(self) -> dict[str, str]:
        """Find README, AGENTS.md, etc. and read them."""
        doc_names = ["AGENTS.md", "CLAUDE.md", "README.md", "pyproject.toml", "package.json"]
        docs = {}
        for base in (Path(self.repo_root), self.cwd):
            for name in doc_names:
                path = base / name
                if path.exists() and str(path) not in docs:
                    try:
                        content = path.read_text(errors="replace")[:1200]
                        docs[str(path)] = content
                    except Exception:
                        pass
        return docs

    def to_prompt_string(self) -> str:
        """Format workspace info for the system prompt."""
        commits = "\n".join(f"  - {c}" for c in self.recent_commits)
        docs = "\n".join(f"  - {k}\n    {v[:200]}..." for k, v in self.project_docs.items())

        return f"""Workspace:
  cwd: {self.cwd}
  repo_root: {self.repo_root}
  branch: {self.branch}
  status: {self.status}
  recent_commits:
{commits}
  project_docs:
{docs}"""