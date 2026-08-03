"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel

from config.model_refs import parse_provider_type
from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import trace_event

from . import dependencies
from .dependencies import get_settings, require_api_key
from .handlers import MessagesHandler, ResponsesHandler, TokenCountHandler
from .model_catalog import build_models_list_response
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.openai_responses import OpenAIResponsesRequest
from .models.responses import ModelsListResponse

router = APIRouter()


def _provider_getter(request: Request, settings: Settings):
    return lambda provider_type: dependencies.resolve_provider(
        provider_type, app=request.app
    )


def get_messages_handler(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> MessagesHandler:
    """Build the Claude Messages product handler for route handlers."""
    return MessagesHandler(
        settings,
        provider_getter=_provider_getter(request, settings),
        token_counter=get_token_count,
    )


def get_responses_handler(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ResponsesHandler:
    """Build the OpenAI Responses product handler for route handlers."""
    return ResponsesHandler(
        settings,
        provider_getter=_provider_getter(request, settings),
    )


def get_token_count_handler(
    settings: Settings = Depends(get_settings),
) -> TokenCountHandler:
    """Build the token-count product handler for route handlers."""
    return TokenCountHandler(settings, token_counter=get_token_count)


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    handler: MessagesHandler = Depends(get_messages_handler),
    _auth=Depends(require_api_key),
):
    """Create a message (streaming by default; stream=false gets aggregated JSON)."""
    return await handler.create(request_data)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request_data: OpenAIResponsesRequest,
    handler: ResponsesHandler = Depends(get_responses_handler),
    _auth=Depends(require_api_key),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await handler.create(request_data)


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_api_key)):
    """Respond to OpenAI Responses compatibility probes."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    handler: TokenCountHandler = Depends(get_token_count_handler),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return handler.count(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": parse_provider_type(settings.model),
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to unauthenticated local compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    trace_event(stage="ingress", event="api.models.list", source="api")
    provider_runtime = dependencies.maybe_provider_runtime(request.app)
    return build_models_list_response(settings, provider_runtime)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    workflow = getattr(request.app.state, "messaging_workflow", None)
    if not workflow:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await workflow.stop_all_tasks()
    trace_event(
        stage="ingress",
        event="api.cli.stop_via_messaging_workflow",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=messaging_workflow cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}


@router.get("/api/models")
async def api_list_models(settings: Settings = Depends(get_settings)):
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Failed to query Ollama tags API: {}", e)
    return []


@router.get("/api/config")
async def api_get_config(settings: Settings = Depends(get_settings)):
    return {
        "ollama_reasoning_model": settings.ollama_reasoning_model,
        "ollama_coding_model": settings.ollama_coding_model,
        "ollama_voice_model": settings.ollama_voice_model or "",
        "ollama_image_model": settings.ollama_image_model or "",
        "ollama_reasoning_system_prompt": settings.ollama_reasoning_system_prompt,
        "bridge_head_model": settings.bridge_head_model,
        "bridge_coding_model": settings.bridge_coding_model,
        "bridge_tooling_model": settings.bridge_tooling_model,
        "context_recent_turns": settings.context_recent_turns,
        "context_max_result_chars": settings.context_max_result_chars,
        "context_max_write_lines": settings.context_max_write_lines,
        "context_head_recent_turns": settings.context_head_recent_turns,
    }


class ConfigUpdatePayload(BaseModel):
    ollama_reasoning_model: str
    ollama_coding_model: str
    ollama_voice_model: str | None = None
    ollama_image_model: str | None = None
    ollama_reasoning_system_prompt: str
    bridge_head_model: str
    bridge_coding_model: str
    bridge_tooling_model: str
    context_recent_turns: int = 3
    context_max_result_chars: int = 3000
    context_max_write_lines: int = 10
    context_head_recent_turns: int = 5


@router.post("/api/config")
async def api_post_config(payload: ConfigUpdatePayload, request: Request):
    import os

    from config.paths import managed_env_path

    updates = {
        "OLLAMA_REASONING_MODEL": payload.ollama_reasoning_model,
        "OLLAMA_CODING_MODEL": payload.ollama_coding_model,
        "OLLAMA_VOICE_MODEL": payload.ollama_voice_model or "",
        "OLLAMA_IMAGE_MODEL": payload.ollama_image_model or "",
        "OLLAMA_REASONING_SYSTEM_PROMPT": payload.ollama_reasoning_system_prompt,
        "BRIDGE_HEAD_MODEL": payload.bridge_head_model,
        "BRIDGE_CODING_MODEL": payload.bridge_coding_model,
        "BRIDGE_TOOLING_MODEL": payload.bridge_tooling_model,
        "CONTEXT_RECENT_TURNS": str(payload.context_recent_turns),
        "CONTEXT_MAX_RESULT_CHARS": str(payload.context_max_result_chars),
        "CONTEXT_MAX_WRITE_LINES": str(payload.context_max_write_lines),
        "CONTEXT_HEAD_RECENT_TURNS": str(payload.context_head_recent_turns),
    }

    # Update env file
    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    key_to_index = {}
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith("#") and "=" in line_stripped:
            k, _ = line_stripped.split("=", 1)
            key_to_index[k.strip()] = i

    for k, value in updates.items():
        # Escape value
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        line_content = f'{k}="{escaped}"'
        if k in key_to_index:
            lines[key_to_index[k]] = line_content
        else:
            lines.append(line_content)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update active environment variables
    for k, value in updates.items():
        os.environ[k] = value

    # Clear settings cache and reload
    from config.settings import clear_settings_cache, get_settings

    clear_settings_cache()
    new_settings = get_settings()

    # Safely clean up old provider runtime and initialize new one
    old_runtime = getattr(request.app.state, "provider_runtime", None)
    if old_runtime:
        try:
            await old_runtime.cleanup()
        except Exception as e:
            logger.warning("Failed to clean up old provider runtime: {}", e)

    from providers.runtime import ProviderRuntime

    new_runtime = ProviderRuntime(new_settings)
    request.app.state.provider_runtime = new_runtime
    new_runtime.start_model_list_refresh()

    return {"status": "ok"}


class InstructionsUpdatePayload(BaseModel):
    instructions: str


@router.get("/api/bridge/instructions")
async def api_get_bridge_instructions():
    from config.paths import config_dir_path
    from providers.base import DEFAULT_INSTRUCTIONS

    path = config_dir_path() / "instructions.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")
    return {"instructions": path.read_text(encoding="utf-8")}


@router.post("/api/bridge/instructions")
async def api_post_bridge_instructions(payload: InstructionsUpdatePayload):
    from config.paths import config_dir_path

    path = config_dir_path() / "instructions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.instructions, encoding="utf-8")
    return {"status": "ok"}


class PullModelPayload(BaseModel):
    name: str


@router.post("/api/pull")
async def api_pull_model(
    payload: PullModelPayload, settings: Settings = Depends(get_settings)
):
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/pull",
                json={"name": payload.name, "stream": False},
                timeout=300.0,
            )
            if resp.status_code == 200:
                return {
                    "status": "success",
                    "message": f"Successfully pulled {payload.name}",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ollama returned status {resp.status_code}: {resp.text}",
                }
    except Exception as e:
        logger.error("Failed to pull model: {}", e)
        return {"status": "error", "message": str(e)}


class RoleConfigUpdatePayload(BaseModel):
    planner_model: str | None = None
    caller_model: str | None = None
    coder_model: str | None = None
    sequential_unload: bool | None = None


@router.get("/api/ollama/models")
async def api_get_ollama_models(settings: Settings = Depends(get_settings)):
    """Return list of locally installed models from Ollama."""
    from core.collab_engine import CollabEngine, RoleModelsConfig

    config = RoleModelsConfig(
        planner_model=settings.ollama_planner_model,
        caller_model=settings.ollama_caller_model,
        coder_model=settings.ollama_coder_model,
        sequential_unload=settings.ollama_sequential_unload,
        base_url=settings.ollama_base_url,
    )
    engine = CollabEngine(config)
    models = await engine.list_available_models()
    return {"models": models}


@router.get("/api/config/roles")
async def api_get_role_config(settings: Settings = Depends(get_settings)):
    """Return active model assignments for Planner, Caller, and Coder roles."""
    return {
        "planner_model": settings.ollama_planner_model,
        "caller_model": settings.ollama_caller_model,
        "coder_model": settings.ollama_coder_model,
        "sequential_unload": settings.ollama_sequential_unload,
    }


@router.post("/api/config/roles")
async def api_update_role_config(
    payload: RoleConfigUpdatePayload,
    settings: Settings = Depends(get_settings),
):
    """Update model assignments for Planner, Caller, and Coder roles live."""
    if payload.planner_model is not None:
        settings.ollama_planner_model = payload.planner_model
    if payload.caller_model is not None:
        settings.ollama_caller_model = payload.caller_model
    if payload.coder_model is not None:
        settings.ollama_coder_model = payload.coder_model
    if payload.sequential_unload is not None:
        settings.ollama_sequential_unload = payload.sequential_unload
    return {
        "status": "ok",
        "planner_model": settings.ollama_planner_model,
        "caller_model": settings.ollama_caller_model,
        "coder_model": settings.ollama_coder_model,
        "sequential_unload": settings.ollama_sequential_unload,
    }
