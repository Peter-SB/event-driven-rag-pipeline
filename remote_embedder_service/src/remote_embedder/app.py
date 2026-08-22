"""FastAPI app exposing an LM Studio-compatible embedding API.

Five routes make up the contract the pipeline's ``FallbackEmbeddingModel``
drives, in the order it drives them:

  1. ``GET  /v1/models``            reachability probe (fired before every batch)
  2. ``GET  /api/v0/models``        per-model ``state`` -> loaded / not-loaded
  3. ``POST /api/v1/models/load``   {"model": id}, blocks until usable
  4. ``POST /v1/embeddings``        {"model": id, "input": [...]}
  5. ``POST /api/v1/models/unload`` {"instance_id": id}

``/v1/*`` and ``/api/*`` must share an origin: the client derives the native API
base by stripping the path off ``EMBED_REMOTE_URL``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from remote_embedder.config import Settings, settings as default_settings
from remote_embedder.manager import (
    ModelLoadError,
    ModelManager,
    UnknownModelError,
)
from remote_embedder.registry import all_specs, get_spec
from remote_embedder.schemas import (
    EmbeddingItem,
    EmbeddingsRequest,
    EmbeddingsResponse,
    LoadRequest,
    ModelActionResponse,
    NativeModel,
    NativeModelList,
    OpenAIModel,
    OpenAIModelList,
    UnloadRequest,
    Usage,
)

logger = logging.getLogger(__name__)


def get_manager(request: Request) -> ModelManager:
    """Manager dependency — overridden in tests to inject a fake."""
    return request.app.state.manager


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
) -> FastAPI:
    cfg = settings or default_settings
    mgr = manager or ModelManager(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = cfg.resolved_device()
        logger.info(
            "remote-embedder starting (device=%s max_batch=%d idle_unload=%.0fs)",
            device,
            cfg.max_batch_size,
            cfg.idle_unload_s,
        )
        if device != "cuda":
            # The whole point of this service is CUDA embedding; running on CPU
            # means either no GPU is visible to this process or DEVICE was set
            # to something else on purpose. Either way it's worth a loud line
            # rather than only showing up as unexpectedly slow embed calls.
            logger.warning(
                "Running on device=%r, not CUDA — embedding will be slow and "
                "int8 quantisation will be skipped for every model. Check "
                "nvidia-smi and the installed torch build if this is unexpected.",
                device,
            )
        mgr.start_reaper()
        try:
            yield
        finally:
            mgr.stop_reaper()
            mgr.unload()

    app = FastAPI(
        title="remote-embedder",
        version="0.1.0",
        description="CUDA embedding service with an LM Studio-compatible API",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.manager = mgr

    _register_routes(app)
    return app


def _check_auth(cfg: Settings, authorization: str | None) -> None:
    """Enforce the bearer token only when one is configured.

    With no ``API_KEY`` set the header is accepted and ignored, so a client
    that always sends ``Authorization: Bearer ...`` (as the pipeline does when
    ``EMBED_REMOTE_API_KEY`` is populated) works against an open service.
    """
    if not cfg.api_key:
        return
    expected = f"Bearer {cfg.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def _register_routes(app: FastAPI) -> None:
    # --- health -----------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        """Plain liveness for humans and container probes.

        The pipeline uses ``/v1/models`` instead (its ``EMBED_REMOTE_HEALTH_PATH``
        default), which is equally cheap.
        """
        mgr: ModelManager = app.state.manager
        return {"status": "ok", "loaded_model": mgr.loaded_model_id}

    # --- OpenAI-compatible surface ---------------------------------------

    @app.get("/v1/models", response_model=OpenAIModelList)
    async def list_models(
        cfg: Settings = Depends(get_settings),
        authorization: str | None = Header(default=None),
    ) -> OpenAIModelList:
        """Every model this service *can* serve, regardless of load state.

        Intentionally does no work: the client hits this before every single
        batch as its reachability probe.
        """
        _check_auth(cfg, authorization)
        return OpenAIModelList(data=[OpenAIModel(id=spec.id) for spec in all_specs()])

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    async def embeddings(
        body: EmbeddingsRequest,
        cfg: Settings = Depends(get_settings),
        mgr: ModelManager = Depends(get_manager),
        authorization: str | None = Header(default=None),
    ) -> EmbeddingsResponse:
        _check_auth(cfg, authorization)

        texts = [body.input] if isinstance(body.input, str) else list(body.input)
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")
        if get_spec(body.model) is None:
            raise HTTPException(status_code=400, detail=f"unknown model: {body.model}")

        try:
            # Off the event loop: encode blocks for the whole forward pass, and
            # the client's next reachability probe must not queue behind it.
            vectors = await run_in_threadpool(mgr.encode, body.model, texts)
        except UnknownModelError:
            raise HTTPException(status_code=400, detail=f"unknown model: {body.model}")
        except ModelLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        # Positional alignment with `input` is part of the contract — the client
        # zips the response against its own list by index.
        return EmbeddingsResponse(
            model=body.model,
            data=[
                EmbeddingItem(index=i, embedding=vec) for i, vec in enumerate(vectors)
            ],
            usage=_estimate_usage(texts),
        )

    # --- native (LM Studio-shaped) surface --------------------------------

    @app.get("/api/v0/models", response_model=NativeModelList)
    async def native_models(
        cfg: Settings = Depends(get_settings),
        mgr: ModelManager = Depends(get_manager),
        authorization: str | None = Header(default=None),
    ) -> NativeModelList:
        """Registry listing *with load state*.

        Every known model appears here, unloaded ones included with
        ``state: "not-loaded"``. Dropping them would read to the client as
        "this endpoint doesn't have that model" and send the batch to its local
        fallback instead of requesting a load.
        """
        _check_auth(cfg, authorization)
        return NativeModelList(data=[NativeModel(**s) for s in mgr.states()])

    @app.post("/api/v1/models/load", response_model=ModelActionResponse)
    async def load_model(
        body: LoadRequest,
        cfg: Settings = Depends(get_settings),
        mgr: ModelManager = Depends(get_manager),
        authorization: str | None = Header(default=None),
    ) -> ModelActionResponse:
        """Make a model resident. Blocks until it is actually usable.

        Downloads the weights if missing — a first load is a multi-gigabyte
        fetch before anything touches the GPU, which is why the client allows
        ``EMBED_REMOTE_LOAD_TIMEOUT_S`` (600s by default) for this call.

        An optional ``quantization`` overrides the registry default, letting the
        pipeline keep precision in its own EmbedConfig instead of the two sides
        deciding separately. The response reports what was actually applied.
        """
        _check_auth(cfg, authorization)
        if get_spec(body.model) is None:
            raise HTTPException(status_code=404, detail=f"unknown model: {body.model}")
        try:
            await run_in_threadpool(mgr.load, body.model, body.quantization)
        except UnknownModelError:
            raise HTTPException(status_code=404, detail=f"unknown model: {body.model}")
        except ModelLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return ModelActionResponse(
            model=body.model,
            state="loaded",
            quantization=mgr.applied_quantization,
        )

    @app.post("/api/v1/models/unload", response_model=ModelActionResponse)
    async def unload_model(
        body: UnloadRequest,
        cfg: Settings = Depends(get_settings),
        mgr: ModelManager = Depends(get_manager),
        authorization: str | None = Header(default=None),
    ) -> ModelActionResponse:
        """Free a model's VRAM. Used by the pipeline's live tests to force the
        not-loaded precondition, and handy by hand."""
        _check_auth(cfg, authorization)
        target = body.target()
        if target is None:
            raise HTTPException(
                status_code=400, detail="instance_id (or model) is required"
            )
        unloaded = await run_in_threadpool(mgr.unload, target)
        return ModelActionResponse(
            model=target,
            state="not-loaded",
            detail="unloaded" if unloaded else "was not loaded",
        )


def _estimate_usage(texts: list[str]) -> Usage:
    """Best-effort token count. Nothing downstream reads it; it exists so the
    response is shaped like a real OpenAI one for other clients."""
    approx = sum(max(1, len(t) // 4) for t in texts)
    return Usage(prompt_tokens=approx, total_tokens=approx)


app = create_app()
