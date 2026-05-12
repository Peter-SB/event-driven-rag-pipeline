"""Migrate posts from a legacy SQLite database to the event-driven RAG pipeline.

Reads all rows from the SQLite `posts` table, maps them to the legacy sync
wire format, and POSTs them in batches to POST /sync.

Usage:
    python scripts/migrate_sqlite_to_sync.py <sqlite_path> [options]

Examples:
    python scripts/migrate_sqlite_to_sync.py scripts/posts.db
    python scripts/migrate_sqlite_to_sync.py scripts/posts.db --table-name posts_main_test
    python scripts/migrate_sqlite_to_sync.py scripts/posts.db --url http://localhost:8000/sync --batch-size 100
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# SQLite → wire-format mapping
# ---------------------------------------------------------------------------

def _to_iso(value: Any) -> str | None:
    """Ensure datetime-like values are ISO-8601 strings with a UTC timezone."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    # SQLite stores datetimes as strings; append Z if no tz offset present
    if s[-1].isdigit():
        s += "Z"
    return s


def _row_to_post(row: dict) -> dict:
    """Map a SQLite row (column-name dict) to the legacy POST /sync wire format."""
    folder_id = row.get("folderId")
    folder_ids = [folder_id] if folder_id is not None else []

    post: dict[str, Any] = {
        "id": row["id"],
        "redditId": row["redditId"],
        "url": row["url"],
        "title": row["title"],
        "author": row["author"],
        "redditCreatedAt": _to_iso(row.get("redditCreatedAt")),
        "addedAt": _to_iso(row.get("addedAt")),
        "updatedAt": _to_iso(row.get("updatedAt")) or _to_iso(row.get("addedAt")) or "1970-01-01T00:00:00Z",
        "isRead": bool(row.get("isRead", False)),
        "isFavorite": bool(row.get("isFavorite", False)),
        "folderIds": folder_ids,
    }

    # Optional fields — omit if absent to keep the payload clean
    for src, dest in [
        ("bodyText", "bodyText"),
        ("subreddit", "subreddit"),
        ("customTitle", "customTitle"),
        ("customBody", "customBody"),
        ("notes", "notes"),
        ("rating", "rating"),
        ("extraFields", "extraFields"),
        ("bodyMinHash", "bodyMinHash"),
        ("summary", "summary"),
    ]:
        val = row.get(src)
        if val is not None:
            post[dest] = val

    return post


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite_path", help="Path to the SQLite database file")
    parser.add_argument("--url", default="http://localhost:8000/sync", help="Legacy sync endpoint URL (default: http://localhost:8000/sync)")
    parser.add_argument("--table-name", default="posts_sql", help="Legacy table_name field sent to the API (default: posts_sql)")
    parser.add_argument("--sqlite-table", default="posts", help="SQLite table to read from (default: posts)")
    parser.add_argument("--batch-size", type=int, default=30, help="Posts per request (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Read and map rows but do not send requests")
    args = parser.parse_args()

    # --- Read from SQLite --------------------------------------------------
    conn = sqlite3.connect(args.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT * FROM {args.sqlite_table}").fetchall()  # noqa: S608 — table name is user-supplied CLI arg, not user-controlled input
    finally:
        conn.close()

    if not rows:
        print(f"No rows found in table '{args.sqlite_table}'.")
        return

    posts = [_row_to_post(dict(r)) for r in rows]
    print(f"Read {len(posts)} posts from '{args.sqlite_path}' (table: {args.sqlite_table})")

    if args.dry_run:
        print(f"[dry-run] Would send {len(posts)} posts in batches of {args.batch_size} to {args.url}")
        print(f"[dry-run] Sample payload (first post):\n{json.dumps(posts[0], indent=2)}")
        return

    # --- Send in batches ---------------------------------------------------
    total = len(posts)
    sent = succeeded = failed = 0

    for batch_start in range(0, total, args.batch_size):
        batch = posts[batch_start : batch_start + args.batch_size]
        payload = {
            "posts": batch,
            "table_name": args.table_name,
        }

        try:
            response = _post_json(args.url, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            print(f"[batch {batch_start}–{batch_start + len(batch) - 1}] HTTP {exc.code}: {body}", file=sys.stderr)
            failed += len(batch)
            continue
        except Exception as exc:
            print(f"[batch {batch_start}–{batch_start + len(batch) - 1}] Error: {exc}", file=sys.stderr)
            failed += len(batch)
            continue

        results = response.get("results", [])
        batch_ok = sum(1 for r in results if r.get("success"))
        batch_fail = len(results) - batch_ok
        succeeded += batch_ok
        failed += batch_fail
        sent += len(batch)

        statuses = {}
        for r in results:
            statuses[r.get("status", "unknown")] = statuses.get(r.get("status", "unknown"), 0) + 1
        status_str = ", ".join(f"{k}={v}" for k, v in statuses.items())
        print(f"[{sent}/{total}] batch ok={batch_ok} fail={batch_fail}  ({status_str})")

    print(f"\nDone — {succeeded} succeeded, {failed} failed out of {total} posts.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
