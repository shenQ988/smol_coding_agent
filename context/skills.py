"""Skill loader — loads markdown skill files and matches them to queries."""

from pathlib import Path


class SkillStore:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, str] = {}
        self._scan()

    def _scan(self):
        """Scan the skills directory for .md files."""
        if not self.skills_dir.exists():
            return
        for path in self.skills_dir.glob("*.md"):
            self.skills[path.stem] = path.read_text()

    def match(self, user_message: str, max_skills: int = 2) -> list[str]:
        """Find relevant skills based on keywords in the user's message."""
        lower = user_message.lower()
        matched = []

        for name, content in self.skills.items():
            # Simple keyword matching: skill name words appear in the message
            keywords = name.replace("-", " ").replace("_", " ").split()
            if any(kw in lower for kw in keywords):
                matched.append(content)

        return matched[:max_skills]

    def list_skills(self) -> list[str]:
        return list(self.skills.keys())
    
    def get_catalog(self) -> str:
        lines = []
        for name, content in self.skills.items():
            first_line = content.strip().splitlines()[0].strip("# ")
            lines.append(f"- {name}: {first_line}")
        return "\n".join(lines) or "- none"
    
    def get(self, name:str) -> str | None:
        return self.skills.get(name)