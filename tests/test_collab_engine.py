"""Unit tests for CollabEngine and multi-role API configuration endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from config.settings import Settings, get_settings
from core.collab_engine import CollabEngine, CollabRole, RoleModelsConfig


def test_collab_engine_init():
    settings = Settings()
    config = RoleModelsConfig(
        planner_model=settings.ollama_planner_model,
        caller_model=settings.ollama_caller_model,
        coder_model=settings.ollama_coder_model,
        sequential_unload=settings.ollama_sequential_unload,
        base_url=settings.ollama_base_url,
    )
    engine = CollabEngine(config)

    assert engine.get_role_model(CollabRole.PLANNER) == settings.ollama_planner_model
    assert engine.get_role_model(CollabRole.CALLER) == settings.ollama_caller_model
    assert engine.get_role_model(CollabRole.CODER) == settings.ollama_coder_model


def test_collab_engine_build_prompt():
    settings = Settings()
    config = RoleModelsConfig(
        planner_model=settings.ollama_planner_model,
        caller_model=settings.ollama_caller_model,
        coder_model=settings.ollama_coder_model,
        sequential_unload=settings.ollama_sequential_unload,
        base_url=settings.ollama_base_url,
    )
    engine = CollabEngine(config)

    prompt = engine.build_role_prompt(
        CollabRole.PLANNER, "Fix authentication bug", "Step 1 done"
    )
    assert "Lead Planner Agent" in prompt
    assert "Fix authentication bug" in prompt
    assert "Step 1 done" in prompt


@pytest.mark.asyncio
async def test_api_role_config_get_and_post():
    settings = Settings()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    # Test GET /api/config/roles
    get_res = client.get("/api/config/roles")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "planner_model" in data
    assert "caller_model" in data
    assert "coder_model" in data
    assert "sequential_unload" in data

    # Test POST /api/config/roles
    update_payload = {
        "planner_model": "gemma2:27b",
        "caller_model": "qwen2.5-coder:14b",
        "coder_model": "deepseek-coder:6.7b",
        "sequential_unload": True,
    }
    post_res = client.post("/api/config/roles", json=update_payload)
    assert post_res.status_code == 200
    updated_data = post_res.json()
    assert updated_data["planner_model"] == "gemma2:27b"
    assert updated_data["caller_model"] == "qwen2.5-coder:14b"
    assert updated_data["coder_model"] == "deepseek-coder:6.7b"
    assert updated_data["sequential_unload"] is True
