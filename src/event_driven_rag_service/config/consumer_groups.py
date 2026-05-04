"""
Central registry of Redpanda/Kafka consumer group names.

Keeping all consumer group IDs in one place:
  - Makes it easy to audit what topics are consumed and by which group.
  - Prevents typos that would silently create a new group with the wrong offset.
  - Documents the full dispatcher-to-topic subscription map at a glance.

Convention: constant name mirrors the event topic with underscores replacing dots.
Value is the group ID string sent to Redpanda/Postgres event bus.
"""

# ---------------------------------------------------------------------------
# PostDispatcher
# ---------------------------------------------------------------------------

POST_SYNCED = "post-dispatcher-synced"
POST_ANALYSED = "post-dispatcher-analysed"   # wired post-MVP for analysis pipeline

# ---------------------------------------------------------------------------
# ChunkDispatcher
# ---------------------------------------------------------------------------

CHUNKS_CREATED = "chunk-dispatcher-chunks-created"

# ---------------------------------------------------------------------------
# SearchDispatcher  (two subscriptions: job created + query embedded)
# ---------------------------------------------------------------------------

SEARCH_JOB_CREATED    = "search-dispatcher-job-created"
SEARCH_QUERY_EMBEDDED = "search-dispatcher-query-embedded"
