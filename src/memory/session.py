from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Session:
    id: str
    created_at: datetime
    source: str
    jsonl_path: str
    memory_md_path: str


class SessionManager:
    def __init__(self, data_dir: str) -> None:
        self._sessions_dir = Path(data_dir) / "sessions"
        self._flat_dir = Path(data_dir) / "flat"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._flat_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def get_or_create(self, session_id: str, source: str = "unknown") -> Session:
        if session_id in self._cache:
            return self._cache[session_id]

        jsonl_path = str(self._sessions_dir / f"{session_id}.jsonl")
        memory_md_path = str(self._flat_dir / f"{session_id}_memory.md")

        if os.path.exists(jsonl_path):
            # Load existing session metadata from first line
            with open(jsonl_path, "r") as f:
                first_line = f.readline()
            try:
                meta = json.loads(first_line)
                created_at = datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat()))
                source = meta.get("source", source)
            except (json.JSONDecodeError, KeyError):
                created_at = datetime.utcnow()
        else:
            created_at = datetime.utcnow()
            # Write session header
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "type": "session_meta",
                    "session_id": session_id,
                    "source": source,
                    "created_at": created_at.isoformat(),
                }) + "\n")

        session = Session(
            id=session_id,
            created_at=created_at,
            source=source,
            jsonl_path=jsonl_path,
            memory_md_path=memory_md_path,
        )
        self._cache[session_id] = session
        return session

    def append_turn(self, session_id: str, turn: dict[str, Any]) -> None:
        """Append one conversation turn to the session JSONL file."""
        session = self.get_or_create(session_id)
        turn["_ts"] = datetime.utcnow().isoformat()
        with open(session.jsonl_path, "a") as f:
            f.write(json.dumps(turn) + "\n")

    def load_history(self, session_id: str, last_n: int = 20) -> list[dict[str, Any]]:
        """Return the last N conversation turns (user/assistant messages)."""
        session = self.get_or_create(session_id)
        if not os.path.exists(session.jsonl_path):
            return []

        turns = []
        with open(session.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Skip session metadata lines
                if obj.get("type") == "session_meta":
                    continue
                if obj.get("role") in ("user", "assistant"):
                    turns.append(obj)

        return turns[-last_n:] if len(turns) > last_n else turns

    def get_memory_context(self, session_id: str) -> str:
        """Read the flat markdown memory file for this session."""
        session = self.get_or_create(session_id)
        if not os.path.exists(session.memory_md_path):
            return ""
        with open(session.memory_md_path, "r") as f:
            return f.read().strip()

    def write_memory_context(self, session_id: str, content: str) -> None:
        """Overwrite the flat markdown memory file."""
        session = self.get_or_create(session_id)
        with open(session.memory_md_path, "w") as f:
            f.write(content)

    def list_active_sessions(self) -> list[str]:
        """Return all session IDs that have JSONL files."""
        return [
            p.stem
            for p in self._sessions_dir.glob("*.jsonl")
        ]
