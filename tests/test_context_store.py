"""Unit tests for core context store, budget manager, and smart compressor."""

from core.context.budget import TokenBudgetManager, estimate_tokens
from core.context.compressor import smart_compress_history
from core.context.store import SessionContextStore, extract_file_skeleton


def test_extract_file_skeleton_python() -> None:
    code = """import os
import sys

class TestClass:
    def __init__(self):
        print("initializing")
        x = 10
        y = 20

    def compute(self):
        return 42

def top_level_func():
    pass
"""
    skeleton = extract_file_skeleton(code, max_lines=10)
    assert "class TestClass" in skeleton
    assert "def compute" in skeleton
    assert "import os" in skeleton


def test_session_context_store_file_caching() -> None:
    store = SessionContextStore(max_files_per_session=3)
    session_id = "test_session_1"

    store.update_file(session_id, "src/main.py", "def main():\n    pass")
    store.update_file(session_id, "src/utils.py", "def help():\n    pass")

    summary = store.get_active_files_summary(session_id)
    assert "src/main.py" in summary
    assert "src/utils.py" in summary


def test_token_estimate_and_budget() -> None:
    text = "Hello world context test string"
    tokens = estimate_tokens(text)
    assert tokens > 0

    manager = TokenBudgetManager(
        max_tokens_head=1000,
        max_tokens_coding=500,
        max_tokens_tooling=200,
    )
    assert manager.get_budget_for_role("head") == 1000
    assert manager.get_budget_for_role("coding") == 500

    messages = [
        {"role": "user", "content": "Long task step 1 " * 50},
        {"role": "assistant", "content": "Long answer step 1 " * 50},
        {"role": "user", "content": "Short step 2"},
    ]
    fitted = manager.fit_messages_to_budget(messages, "System prompt", max_tokens=100)
    assert len(fitted) <= len(messages)
    assert fitted[-1]["content"] == "Short step 2"


def test_smart_compress_history() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "● <function=view_file>\n<parameter=TargetFile>app.py</parameter>",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[Tool Result: def app():\n    return 'OK']",
                }
            ],
        },
    ]

    store = SessionContextStore()
    compressed = smart_compress_history(
        messages, session_id="sess_test", session_store=store
    )
    assert len(compressed) == 2
    summary = store.get_active_files_summary("sess_test")
    assert "app.py" in summary
