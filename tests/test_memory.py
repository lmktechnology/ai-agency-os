from __future__ import annotations

import json

import pytest

from src.memory.session import SessionManager


def test_create_session(tmp_path):
    sm = SessionManager(str(tmp_path))
    session = sm.get_or_create("test_123", source="cli")

    assert session.id == "test_123"
    assert session.source == "cli"
    import os
    assert os.path.exists(session.jsonl_path)


def test_append_and_load_history(tmp_path):
    sm = SessionManager(str(tmp_path))
    sm.get_or_create("sess_1")

    sm.append_turn("sess_1", {"role": "user", "content": "Hello"})
    sm.append_turn("sess_1", {"role": "assistant", "content": "Hi there"})

    history = sm.load_history("sess_1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"


def test_load_history_respects_last_n(tmp_path):
    sm = SessionManager(str(tmp_path))
    sm.get_or_create("sess_2")

    for i in range(10):
        sm.append_turn("sess_2", {"role": "user", "content": f"Message {i}"})

    history = sm.load_history("sess_2", last_n=3)
    assert len(history) == 3
    assert history[-1]["content"] == "Message 9"


def test_memory_context_read_write(tmp_path):
    sm = SessionManager(str(tmp_path))
    sm.get_or_create("sess_3")

    assert sm.get_memory_context("sess_3") == ""

    sm.write_memory_context("sess_3", "Summary of previous session.")
    assert sm.get_memory_context("sess_3") == "Summary of previous session."


def test_list_active_sessions(tmp_path):
    sm = SessionManager(str(tmp_path))
    sm.get_or_create("a")
    sm.get_or_create("b")
    sm.get_or_create("c")

    sessions = sm.list_active_sessions()
    assert set(sessions) == {"a", "b", "c"}
