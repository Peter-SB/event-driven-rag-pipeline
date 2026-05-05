"""Quick script to test event publishing and consumption."""
import asyncio
import json
import asyncpg
import httpx
from datetime import datetime, UTC


async def main():
    pool = await asyncpg.create_pool("postgresql://rag:rag@localhost:5432/rag")

    # Clean slate
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM event_log")
        await conn.execute("DELETE FROM consumer_offsets")

    # POST a post
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as client:
        payload = {
            "posts": [
                {
                    "id": 999,
                    "redditId": "reddit_999",
                    "externalSource": "reddit",
                    "redditCreatedAt": datetime.now(UTC).isoformat(),
                    "url": "https://reddit.com/r/test/comments/999",
                    "title": "Debug Post",
                    "bodyText": "Debug content for testing event publishing. " * 5,
                    "author": "test_user",
                    "addedAt": datetime.now(UTC).isoformat(),
                    "updatedAt": datetime.now(UTC).isoformat(),
                }
            ],
            "library_id": "e2e",
        }
        r = await client.post("/posts/sync", json=payload)
        print(f"POST status: {r.status_code}")
        print(f"Response: {r.json()}")

    # Check event_log
    async with pool.acquire() as conn:
        events = await conn.fetch("SELECT id, topic, payload FROM event_log ORDER BY id")
        print(f"Events in log: {len(events)}")
        for e in events:
            p = json.loads(e["payload"])
            print(f"  id={e['id']} topic={e['topic']} post_id={p.get('post_id')}")

        offsets = await conn.fetch("SELECT * FROM consumer_offsets")
        print(f"Offsets: {[(r['consumer_group'], r['topic'], r['last_id']) for r in offsets]}")

    await pool.close()


asyncio.run(main())
