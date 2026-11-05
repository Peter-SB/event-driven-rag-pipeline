"""HTMX search UI — serves the search page and result pages."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path

import asyncpg
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from event_driven_rag_service.api.recommend import run_recommendation
from event_driven_rag_service.api.search import SearchResultItem
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.repository.recommendation_repository import FeedbackRepository
from event_driven_rag_service.services.playlist_recommendation import EmptyPlaylistError, RecommendParams
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"])


def _parse_results(raw) -> list[SearchResultItem]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [SearchResultItem(**r) for r in raw]


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "search.html", {
        "chunk_types": list(EMBED_CONFIGS.keys()),
    })


@router.post("/search/submit", response_class=HTMLResponse)
async def submit_search(
    request: Request,
    query: str = Form(...),
    library_id: str = Form("main"),
    chunk_type: str = Form("body"),
    k: int = Form(10),
    rerank: bool = Form(False),
) -> HTMLResponse:
    if not re.match(r"^[a-z][a-z0-9_]*$", library_id):
        return HTMLResponse(
            '<tr><td colspan="7" class="error-row">Invalid library_id — must match [a-z][a-z0-9_]*</td></tr>',
            status_code=422,
        )
    if chunk_type not in EMBED_CONFIGS:
        return HTMLResponse(
            f'<tr><td colspan="7" class="error-row">Invalid chunk_type — must be one of {list(EMBED_CONFIGS.keys())}</td></tr>',
            status_code=422,
        )

    job_repo: SearchJobRepository = request.app.state.search_job_repo
    event_bus = request.app.state.event_bus

    # "Advanced (rerank)" checkbox: unchecked (the default) never reaches the
    # rerank path. Checked uses the "default" profile from RERANK_CONFIGS.
    rerank_profile = "default" if rerank else None

    embed_cfg = EMBED_CONFIGS[chunk_type]
    chunks_table = build_chunk_table_name(f"posts_{library_id}", chunk_type, embed_cfg.primary)

    chunk_repo = ChunkRepository(request.app.state.pool)
    if not await chunk_repo.table_exists(chunks_table):
        return HTMLResponse(
            f'<tr><td colspan="7" class="error-row">No chunk table found for library '
            f'{library_id!r} and chunk_type {chunk_type!r} — has this library been synced yet?</td></tr>',
            status_code=404,
        )

    job_id = await job_repo.create_job(
        query=query,
        k=k,
        embedding_profile=embed_cfg.primary,
        chunks_table=chunks_table,
        library_id=library_id,
        rerank=rerank,
        rerank_profile=rerank_profile,
    )

    event = SearchJobCreatedEvent(
        query_job_id=job_id,
        query=query,
        embedding_profile=embed_cfg.primary,
        rerank=rerank,
        rerank_profile=rerank_profile,
    )
    await event_bus.publish(event.event_type, event.to_dict())

    logger.info("ui: created search job %s query=%r library=%s", job_id, query, library_id)

    return templates.TemplateResponse(request, "partials/job_row.html", {
        "job_id": job_id,
        "query": query,
        "library_id": library_id,
        "chunk_type": chunk_type,
        "k": k,
        "status": "embedding",
        "created_at": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "result_count": None,
        "rerank": rerank,
    })


@router.get("/search/{job_id}/status", response_class=HTMLResponse)
async def job_status(job_id: str, request: Request) -> HTMLResponse:
    job_repo: SearchJobRepository = request.app.state.search_job_repo
    job = await job_repo.get_job(job_id)

    if job is None:
        return HTMLResponse(
            f'<tr id="job-{job_id}"><td colspan="7" class="error-row">Job {job_id} not found</td></tr>'
        )

    created_at = job.get("created_at")
    if created_at and hasattr(created_at, "strftime"):
        created_at = created_at.strftime("%H:%M:%S")

    raw_results = job.get("results")
    result_count: int | None = None
    if raw_results:
        if isinstance(raw_results, str):
            raw_results = json.loads(raw_results)
        result_count = len(raw_results)

    return templates.TemplateResponse(request, "partials/job_row.html", {
        "job_id": job_id,
        "query": job["query"],
        "library_id": job.get("library_id", ""),
        "chunk_type": job.get("embedding_profile", ""),
        "k": job["k"],
        "status": job["status"],
        "created_at": created_at,
        "result_count": result_count,
        "rerank": bool(job.get("rerank", False)),
    })


@router.get("/search/{job_id}/poll", response_class=HTMLResponse)
async def poll_results(job_id: str, request: Request) -> HTMLResponse:
    """Returns the #results-content div; replaces itself via htmx until terminal status."""
    job_repo: SearchJobRepository = request.app.state.search_job_repo
    job = await job_repo.get_job(job_id)

    if job is None:
        return HTMLResponse(f'<div id="results-content"><p>Job {job_id} not found.</p></div>')

    results: list[SearchResultItem] = []
    if job.get("results"):
        results = _parse_results(job["results"])

    return templates.TemplateResponse(request, "partials/results_content.html", {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "results": results,
        "library_id": job.get("library_id", ""),
    })


@router.get("/search/{job_id}", response_class=HTMLResponse)
async def results_page(job_id: str, request: Request) -> HTMLResponse:
    job_repo: SearchJobRepository = request.app.state.search_job_repo
    job = await job_repo.get_job(job_id)

    if job is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)

    results: list[SearchResultItem] = []
    if job.get("results"):
        results = _parse_results(job["results"])

    created_at = job.get("created_at")
    if created_at and hasattr(created_at, "strftime"):
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    return templates.TemplateResponse(request, "results.html", {
        "job_id": job_id,
        "query": job["query"],
        "status": job["status"],
        "error": job.get("error"),
        "results": results,
        "created_at": created_at,
        "k": job["k"],
        "library_id": job.get("library_id", ""),
    })


@router.get("/post/{library_id}/{post_id}", response_class=HTMLResponse)
async def post_page(library_id: str, post_id: int, request: Request) -> HTMLResponse:
    if not re.match(r"^[a-z][a-z0-9_]*$", library_id):
        return HTMLResponse("<h1>Invalid library_id — must match [a-z][a-z0-9_]*</h1>", status_code=422)

    post_repo = PostRepository(request.app.state.pool)
    post_table = f"posts_{library_id}"

    try:
        post = await post_repo.fetch(post_id, post_table)
    except asyncpg.exceptions.UndefinedTableError:
        post = None

    if post is None:
        return HTMLResponse(
            f"<h1>Post {post_id} not found in library {library_id!r}</h1>", status_code=404
        )

    created_at = post.external_created_at
    if created_at and hasattr(created_at, "strftime"):
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    return templates.TemplateResponse(request, "post.html", {
        "post_id": post.post_id,
        "library_id": library_id,
        "title": post.title,
        "summary": post.summary,
        "body_text": post.body_text,
        "url": post.url,
        "created_at": created_at,
    })


# ---------------------------------------------------------------------------
# Playlist recommendations
# ---------------------------------------------------------------------------

def _recommend_error(message: str, status_code: int = 422) -> HTMLResponse:
    return HTMLResponse(f'<div class="rec-error">{html_escape(message)}</div>', status_code=status_code)


@router.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "recommend.html", {
        "chunk_types": list(EMBED_CONFIGS.keys()),
    })


@router.post("/recommend/submit", response_class=HTMLResponse)
async def recommend_submit(
    request: Request,
    library_id: str = Form(...),
    chunk_type: str = Form("summary_title"),
    folder_id: str = Form(""),
    post_ids: str = Form(""),
    k: int = Form(10),
    diversity: float = Form(0.3),
) -> HTMLResponse:
    if not re.match(r"^[a-z][a-z0-9_]*$", library_id):
        return _recommend_error("Invalid library id — must match [a-z][a-z0-9_]*")
    if chunk_type not in EMBED_CONFIGS:
        return _recommend_error(f"Invalid chunk type — must be one of {list(EMBED_CONFIGS)}")

    ids = [int(x) for x in re.findall(r"\d+", post_ids)]
    folder = int(folder_id) if folder_id.strip().isdigit() else None
    if ids:
        folder = None  # an explicit list wins over a folder
    elif folder is None:
        return _recommend_error("Enter a folder number or a list of post ids.")

    try:
        key, result = await run_recommendation(
            request.app.state.pool,
            library_id=library_id,
            chunk_type=chunk_type,
            folder_id=folder,
            post_ids=ids or None,
            params=RecommendParams(k=max(1, min(k, 100)), diversity=max(0.0, min(diversity, 1.0))),
        )
    except ChunkTableNotFoundError as exc:
        return _recommend_error(f"Table not found: {exc} — has this library been synced?", 404)
    except EmptyPlaylistError as exc:
        return _recommend_error(str(exc))

    return templates.TemplateResponse(request, "partials/recommend_results.html", {
        "library_id": library_id,
        "playlist_key": key,
        "result": result,
    })


@router.post("/recommend/feedback", response_class=HTMLResponse)
async def recommend_feedback(
    request: Request,
    library_id: str = Form(...),
    playlist_key: str = Form(...),
    post_id: int = Form(...),
    verdict: str = Form(...),
    title: str = Form(""),
) -> HTMLResponse:
    """Record yes/no/clear for one card and return the card's new state."""
    error = None
    if verdict not in ("yes", "no", "clear") or not re.match(r"^[a-z][a-z0-9_]*$", library_id):
        return _recommend_error("Bad feedback request", 422)
    try:
        await FeedbackRepository(request.app.state.pool).set_verdict(library_id, playlist_key, post_id, verdict)
    except Exception as exc:  # e.g. read-only database
        logger.warning("recommend feedback failed: %s", exc)
        error = "Could not save feedback (is the database read-only?)"
        verdict = "clear"
    return templates.TemplateResponse(request, "partials/recommend_feedback.html", {
        "library_id": library_id,
        "playlist_key": playlist_key,
        "post_id": post_id,
        "title": title,
        "verdict": verdict,
        "error": error,
    })
