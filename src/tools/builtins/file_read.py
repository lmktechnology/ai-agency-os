from __future__ import annotations

import os
from pathlib import Path


def file_read(path: str, max_chars: int = 4000) -> dict:
    """
    Read a file from the skills/ or memory/ directories.
    Restricted to the project data directories for safety.
    """
    # Resolve to absolute path and validate it stays within allowed dirs
    abs_path = Path(path).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    allowed_dirs = [
        project_root / "skills",
        project_root / "memory",
        project_root / "agents",
    ]

    if not any(str(abs_path).startswith(str(d)) for d in allowed_dirs):
        return {"error": f"Access denied: {path} is outside allowed directories"}

    if not abs_path.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = abs_path.read_text(encoding="utf-8")
        truncated = len(content) > max_chars
        return {
            "path": str(abs_path),
            "content": content[:max_chars],
            "truncated": truncated,
            "total_chars": len(content),
        }
    except Exception as e:
        return {"error": str(e)}
