"""Quick script to inspect event_log and consumer_offsets state."""
import asyncio
import json
import asyncpg


async def main():
    pool = await asyncpg.create_pool("postgresql://rag:rag@localhost:5432/rag")
    async with pool.acquire() as conn:
        events = await conn.fetch("SELECT id, topic, payload FROM event_log ORDER BY id")
        print(f"Event count: {len(events)}")
        for e in events:
            p = json.loads(e["payload"])
            print(f"  id={e['id']} topic={e['topic']} post_id={p.get('post_id')} keys={list(p.keys())}")

        offsets = await conn.fetch("SELECT * FROM consumer_offsets")
        print(f"Offsets: {[(r['consumer_group'], r['topic'], r['last_id']) for r in offsets]}")

        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'posts%'"
        )
        print(f"Post/chunk tables: {[r['table_name'] for r in tables]}")

    await pool.close()


asyncio.run(main())
