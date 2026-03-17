from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillLoader:
    def __init__(self, skills_dir: str) -> None:
        self._skills_dir = Path(skills_dir)

    def load(self, name: str) -> str:
        """Read skills/{name}.md and return its content as a string."""
        path = self._skills_dir / f"{name}.md"
        if not path.exists():
            logger.warning("Skill file not found: %s", path)
            return ""
        return path.read_text(encoding="utf-8").strip()

    def load_many(self, names: list[str]) -> str:
        """Load multiple skills and concatenate them."""
        parts = []
        for name in names:
            content = self.load(name)
            if content:
                parts.append(f"## Skill: {name}\n{content}")
        return "\n\n".join(parts)
