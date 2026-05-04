Ready to code?                                                                                                                                                             
                                                                                                                                                                            
 Here is Claude's plan:                                                                                                                                                     
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: MVP — Sync + Embed Pipeline (No Search/Inference)                                                                                                                    
                                                                                                                                                                          
 Context

 The sync → chunk → embed pipeline is largely coded but has critical runtime bugs that prevent it from running end-to-end. Dispatcher processes have no entrypoints and are
  missing from docker-compose. The goal is a working MVP: POST /posts/sync flows through to stored chunk embeddings in Postgres, with clean code, solid tests, and
 finalised docs.

 ---
 Part 1 — Design Clarifications

 Open questions to resolve before or during implementation:

 1. PostFetcher Protocol mismatch — ChunkPostHandler.PostFetcher declares fetch(post_id, table) but PostRepository.fetch(post_id) takes no table. Posts live in a single
 table, so the table param on the Protocol appears spurious. Recommended fix: remove table from the Protocol and leave PostRepository.fetch as-is. Confirm before touching.
 2. Chunk table auto-creation responsibility — nobody currently calls chunk_repo.ensure_table() in any entrypoint. GPU entrypoints create ChunkRepository(pool) with no
 vector_dim. Recommended: GPU entrypoint should call ensure_table(table, vector_dim) per configured EmbedConfig at startup.
 3. analysis_text in ChunkTask — there is a # todo: remove and dont carry text in the event comment. For MVP (no inference) this is irrelevant; defer explicitly and note
 in CONTEXT.md.
 4. extra="forbid" in BaseEvent — the ??  comment on line 36 of base_event.py. Decision needed: forbid (strict, catches bugs) vs allow (forward-compatible). Recommend
 extra="forbid" for MVP — easier to relax later than tighten.

 ---
 Part 2 — Critical Bug Fixes (MVP can't run without these)

 Bug 1: post_repository.py — broken psycopg2 contamination

 File: src/event_driven_rag_service/repository/post_repository.py

 The file has two incompatible implementations merged. Lines ~164–436 contain unreachable psycopg2-based code with undefined variables (POSTGRES_URI, COLUMNS, DictCursor).
  Worse, the mark_embedded async method has orphaned error-handling code inside its body referencing non-existent private methods (self._upsert_post_attempt,
 self._ensure_table). This will raise AttributeError on any exception path.

 Fix: Delete everything after the closing of the mark_embedded async block (lines ~163 onwards). Keep only the clean asyncpg class.

 ---
 Bug 2: PostFetcher Protocol/PostRepository mismatch

 File: src/event_driven_rag_service/handlers/chunk_handler.py

 PostFetcher.fetch(post_id, table) vs PostRepository.fetch(post_id) — runtime TypeError when CPU worker processes its first task.

 Fix: Remove table: str from the PostFetcher Protocol. Update the handler call site accordingly. (Posts always live in one table — the param is spurious.)

 ---
 Bug 3: No dispatcher entrypoint

 File to create: src/event_driven_rag_service/worker/entrypoints/dispatcher.py

 Dispatchers bridge the event bus to RabbitMQ. Workers have python -m ...cpu and python -m ...gpu entrypoints; dispatchers have nothing. They cannot be run as standalone
 processes.

 Fix: Create a single combined entrypoint (python -m event_driven_rag_service.worker.entrypoints.dispatcher) that runs PostDispatcher and ChunkDispatcher together —
 simpler for homelab, matches the single-process constraint.

 ---
 Bug 4: No dispatcher service in docker-compose.yml

 Fix: Add one dispatcher service to docker-compose.yml running the combined dispatcher entrypoint, with EVENT_BUS, DB_URL, RABBITMQ_URL, and REDPANDA_SERVERS env vars.

 ---
 Bug 5: Chunk table auto-creation missing

 File: src/event_driven_rag_service/worker/entrypoints/gpu.py

 GPU entrypoint creates ChunkRepository(pool) without calling ensure_table(). If chunk tables don't exist, workers will fail on first embed task.

 Fix: In GPU entrypoint startup, iterate EMBED_CONFIGS and call chunk_repo.ensure_table(table_name, vector_dim) for each config.

 ---
 Part 3 — Dead Code Cleanup

 ┌──────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
 │                     Item                     │                                       Action                                       │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../worker/handlers/chunk_handler.py     │ Delete — orphan duplicate, nothing imports it                                      │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../repository/search_job_repository.py  │ Delete — wrong project codebase, broken imports, nothing uses it                   │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../data_models/post.py lines 65–93      │ Delete commented-out old model with reddit_id fields                               │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../api/sync.py lines 113–124            │ Delete accidental JS/TS code snippet                                               │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../dispatchers/__init__.py              │ Export all four dispatchers, not just PostDispatcher                               │
 ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ src/.../infrastructure/base_event.py line 23 │ Fix typo "impodence" → "idempotence"; remove ?? comment — confirmed extra="forbid" │
 └──────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

 ---
 Part 4 — Documentation Finalization

 ┌──────────────────────┬────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │         Doc          │   Status   │                                                              Action                                                              │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ CONTEXT.md           │ Good       │ Fix "Out of Scope" section — EmbeddingDispatcher is listed as out of scope but is implemented. Clarify which parts are post-MVP. │
 │                      │            │  Add note about analysis_text deferral. Add dispatcher process deployment note.                                                  │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Readme.md            │ Incomplete │ Fix incomplete sentence at line 75 (cuts off mid-thought on testing). Tidy pipeline/notes section.                               │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ docs/Architecture.md │ Partial    │ Complete the Workers section (currently cut off). Add dispatcher deployment section. Fix the Full Pipeline Map dispatcher list   │
 │                      │            │ (shows only PostDispatcher + EmbeddingDispatcher but ChunkDispatcher and SearchDispatcher are also implemented).                 │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ docs/Testing.md      │ Good       │ Add a fixture table row for the mock embedding model. Update test lists section to reflect current actual tests vs planned.      │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ docs/Glossary.md     │ Stub       │ Add Dispatcher, Task, Event Bus entries (currently missing).                                                                     │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ TESTS.md             │ Missing    │ Create — referenced in Readme.md but doesn't exist. Cover directory structure, fixture summary, and running instructions.        │
 ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ CLAUDE.md            │ Missing    │ Create — project context for AI assistant: architecture overview, key design decisions, gotchas (the two-layer design, chunk     │
 │                      │            │ table naming, PostFetcher protocol).                                                                                             │
 └──────────────────────┴────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 ---
 Part 5 — Tests

 Missing Unit Tests

 tests/unit/test_embed_handler.py (highest priority — no unit tests for EmbedHandler at all)
 - test_embed_handler_fetches_texts_and_stores_embeddings
 - test_embed_handler_skips_empty_chunk_list
 - test_embed_handler_publishes_embedding_completed_event
 - Uses: FakeChunkFetcher, FakeEmbeddingStore, deterministic mock model

 ---
 Missing Integration Tests

 tests/integration/test_gpu_worker.py
 - test_gpu_worker_embeds_chunks_and_persists_vectors — real Postgres, MOCK_EMBEDDINGS=1
 - test_gpu_worker_creates_chunk_table_if_missing — verifies startup ensure_table call (after Bug 5 fix)

 ---
 Missing E2E Test

 tests/e2e/test_ingest_pipeline.py — requires dispatcher services in docker-compose first
 - test_sync_post_triggers_chunk_and_embed_pipeline — POST sync → poll for chunks → poll for embeddings
 - Uses MOCK_EMBEDDINGS=1 on GPU worker; polls DB up to ~10s with backoff

 ---
 Existing Tests to Verify After Fixes

 ┌────────────────────────────────────────────┬───────────────────────────────────────────────┐
 │                    Test                    │                     Check                     │
 ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ tests/unit/test_cpu_worker.py              │ Passes with PostFetcher Protocol fix          │
 ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ tests/unit/test_chunk_dispatcher.py        │ Still correct after any dispatcher cleanup    │
 ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ tests/integration/test_chunk_repository.py │ Covers update_embeddings and ensure_table     │
 ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ tests/e2e/test_sync_api.py                 │ Still passes after post_repository.py cleanup │
 └────────────────────────────────────────────┴───────────────────────────────────────────────┘

 ---
 Execution Order

 1. Design clarifications (confirm PostFetcher fix, extra="forbid" decision)

● Updated plan
  ⎿  /plan to preview

● Updated plan
  ⎿  /plan to preview

● Updated plan
  ⎿  /plan to preview

● User approved Claude's plan               
  ⎿  Plan saved to: C:\Users\Peter\.claude\plans\agile-beaming-lynx.md · /plan to edit    
     Plan: MVP — Sync + Embed Pipeline (No Search/Inference)
                                                                                                                                                                            
     Context
                                                                                                                                                                            
     The sync → chunk → embed pipeline is largely coded but has critical runtime bugs that prevent it from running end-to-end. Dispatcher processes have no entrypoints and
     are missing from docker-compose. The goal is a working MVP: POST /posts/sync flows through to stored chunk embeddings in Postgres, with clean code, solid tests, and
     finalised docs.

     ---
     Part 1 — Design Clarifications

     Open questions to resolve before or during implementation:

     1. PostFetcher Protocol mismatch — ChunkPostHandler.PostFetcher declares fetch(post_id, table) but PostRepository.fetch(post_id) takes no table. Posts live in a single
      table, so the table param on the Protocol appears spurious. Recommended fix: remove table from the Protocol and leave PostRepository.fetch as-is. Confirm before
     touching.
     2. Chunk table auto-creation responsibility — nobody currently calls chunk_repo.ensure_table() in any entrypoint. GPU entrypoints create ChunkRepository(pool) with no
     vector_dim. Recommended: GPU entrypoint should call ensure_table(table, vector_dim) per configured EmbedConfig at startup.
     3. analysis_text in ChunkTask — there is a # todo: remove and dont carry text in the event comment. For MVP (no inference) this is irrelevant; defer explicitly and
     note in CONTEXT.md.
     4. extra="forbid" in BaseEvent — the ??  comment on line 36 of base_event.py. Decision needed: forbid (strict, catches bugs) vs allow (forward-compatible). Recommend
     extra="forbid" for MVP — easier to relax later than tighten.

     ---
     Part 2 — Critical Bug Fixes (MVP can't run without these)

     Bug 1: post_repository.py — broken psycopg2 contamination

     File: src/event_driven_rag_service/repository/post_repository.py

     The file has two incompatible implementations merged. Lines ~164–436 contain unreachable psycopg2-based code with undefined variables (POSTGRES_URI, COLUMNS,
     DictCursor). Worse, the mark_embedded async method has orphaned error-handling code inside its body referencing non-existent private methods
     (self._upsert_post_attempt, self._ensure_table). This will raise AttributeError on any exception path.

     Fix: Delete everything after the closing of the mark_embedded async block (lines ~163 onwards). Keep only the clean asyncpg class.

     ---
     Bug 2: PostFetcher Protocol/PostRepository mismatch

     File: src/event_driven_rag_service/handlers/chunk_handler.py

     PostFetcher.fetch(post_id, table) vs PostRepository.fetch(post_id) — runtime TypeError when CPU worker processes its first task.

     Fix: Remove table: str from the PostFetcher Protocol. Update the handler call site accordingly. (Posts always live in one table — the param is spurious.)

     ---
     Bug 3: No dispatcher entrypoint

     File to create: src/event_driven_rag_service/worker/entrypoints/dispatcher.py

     Dispatchers bridge the event bus to RabbitMQ. Workers have python -m ...cpu and python -m ...gpu entrypoints; dispatchers have nothing. They cannot be run as
     standalone processes.

     Fix: Create a single combined entrypoint (python -m event_driven_rag_service.worker.entrypoints.dispatcher) that runs PostDispatcher and ChunkDispatcher together —
     simpler for homelab, matches the single-process constraint.

     ---
     Bug 4: No dispatcher service in docker-compose.yml

     Fix: Add one dispatcher service to docker-compose.yml running the combined dispatcher entrypoint, with EVENT_BUS, DB_URL, RABBITMQ_URL, and REDPANDA_SERVERS env vars.

     ---
     Bug 5: Chunk table auto-creation missing

     File: src/event_driven_rag_service/worker/entrypoints/gpu.py

     GPU entrypoint creates ChunkRepository(pool) without calling ensure_table(). If chunk tables don't exist, workers will fail on first embed task.

     Fix: In GPU entrypoint startup, iterate EMBED_CONFIGS and call chunk_repo.ensure_table(table_name, vector_dim) for each config.

     ---
     Part 3 — Dead Code Cleanup

     ┌──────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
     │                     Item                     │                                       Action                                       │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../worker/handlers/chunk_handler.py     │ Delete — orphan duplicate, nothing imports it                                      │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../repository/search_job_repository.py  │ Delete — wrong project codebase, broken imports, nothing uses it                   │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../data_models/post.py lines 65–93      │ Delete commented-out old model with reddit_id fields                               │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../api/sync.py lines 113–124            │ Delete accidental JS/TS code snippet                                               │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../dispatchers/__init__.py              │ Export all four dispatchers, not just PostDispatcher                               │
     ├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
     │ src/.../infrastructure/base_event.py line 23 │ Fix typo "impodence" → "idempotence"; remove ?? comment — confirmed extra="forbid" │
     └──────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

     ---
     Part 4 — Documentation Finalization

     ┌──────────────────────┬────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┐
     │         Doc          │   Status   │                                                              Action
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ CONTEXT.md           │ Good       │ Fix "Out of Scope" section — EmbeddingDispatcher is listed as out of scope but is implemented. Clarify which parts are post-MVP.
     │
     │                      │            │  Add note about analysis_text deferral. Add dispatcher process deployment note.
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ Readme.md            │ Incomplete │ Fix incomplete sentence at line 75 (cuts off mid-thought on testing). Tidy pipeline/notes section.
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ docs/Architecture.md │ Partial    │ Complete the Workers section (currently cut off). Add dispatcher deployment section. Fix the Full Pipeline Map dispatcher list
     │
     │                      │            │ (shows only PostDispatcher + EmbeddingDispatcher but ChunkDispatcher and SearchDispatcher are also implemented).
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ docs/Testing.md      │ Good       │ Add a fixture table row for the mock embedding model. Update test lists section to reflect current actual tests vs planned.
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ docs/Glossary.md     │ Stub       │ Add Dispatcher, Task, Event Bus entries (currently missing).
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ TESTS.md             │ Missing    │ Create — referenced in Readme.md but doesn't exist. Cover directory structure, fixture summary, and running instructions.
     │
     ├──────────────────────┼────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ CLAUDE.md            │ Missing    │ Create — project context for AI assistant: architecture overview, key design decisions, gotchas (the two-layer design, chunk
     │
     │                      │            │ table naming, PostFetcher protocol).
     │
     └──────────────────────┴────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┘

     ---
     Part 5 — Tests

     Missing Unit Tests

     tests/unit/test_embed_handler.py (highest priority — no unit tests for EmbedHandler at all)
     - test_embed_handler_fetches_texts_and_stores_embeddings
     - test_embed_handler_skips_empty_chunk_list
     - test_embed_handler_publishes_embedding_completed_event
     - Uses: FakeChunkFetcher, FakeEmbeddingStore, deterministic mock model

     ---
     Missing Integration Tests

     tests/integration/test_gpu_worker.py
     - test_gpu_worker_embeds_chunks_and_persists_vectors — real Postgres, MOCK_EMBEDDINGS=1
     - test_gpu_worker_creates_chunk_table_if_missing — verifies startup ensure_table call (after Bug 5 fix)

     ---
     Missing E2E Test

     tests/e2e/test_ingest_pipeline.py — requires dispatcher services in docker-compose first
     - test_sync_post_triggers_chunk_and_embed_pipeline — POST sync → poll for chunks → poll for embeddings
     - Uses MOCK_EMBEDDINGS=1 on GPU worker; polls DB up to ~10s with backoff

     ---
     Existing Tests to Verify After Fixes

     ┌────────────────────────────────────────────┬───────────────────────────────────────────────┐
     │                    Test                    │                     Check                     │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
     │ tests/unit/test_cpu_worker.py              │ Passes with PostFetcher Protocol fix          │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
     │ tests/unit/test_chunk_dispatcher.py        │ Still correct after any dispatcher cleanup    │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
     │ tests/integration/test_chunk_repository.py │ Covers update_embeddings and ensure_table     │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────┤
     │ tests/e2e/test_sync_api.py                 │ Still passes after post_repository.py cleanup │
     └────────────────────────────────────────────┴───────────────────────────────────────────────┘

     ---
     Execution Order

     1. Design clarifications (confirm PostFetcher fix, extra="forbid" decision)
     2. Bug fixes (1 → 2 → 3 → 4 → 5) — pipeline must run before testing it
     3. Dead code cleanup
     4. Unit test for EmbedHandler
     5. Integration test for GPU worker
     6. Documentation finalization
     7. E2E pipeline test (after docker-compose has dispatcher services)

● Perfect. Plan is approved. Let me create a task list to track the work.
                                                                                          
● Creating a task list to track the MVP work:
                                                                                                                                                                            
● Perfect. Your plan is ready to execute. Here's the summary:
                                                                                                                                                                            
  MVP Plan Summary                                                                                                                                                        
                                                                                                                                                                            
  Goal: Working sync → chunk → embed pipeline with clean code, solid tests, and finalised docs.                                                                             
                                                                                                                                                                            
  Key Decisions Confirmed:                                                                                                                                                  
  - Remove table param from PostFetcher Protocol (posts always in one table)                                                                                                
  - Single combined dispatcher.py entrypoint running PostDispatcher + ChunkDispatcher                                                                                       
  - Use extra="forbid" on BaseEvent (strict validation)                                                                                                                     
                                                                                                                                                                          
  ---
  Work Breakdown: 10 Tasks

  🔴 Critical Bug Fixes (5 tasks — must complete before MVP runs)

  1. Fix post_repository.py — remove ~270 lines of unreachable psycopg2 dead code
  2. Fix PostFetcher Protocol mismatch — remove spurious table param
  3. Create dispatcher entrypoint — single combined process for both dispatchers
  4. Add dispatcher service to docker-compose.yml
  5. Add chunk table auto-creation to GPU entrypoint startup

  🟡 Cleanup & Tests (5 tasks)

  6. Remove dead code — orphan duplicate handler, wrong-project search repo, old comments, JS snippet
  7. Unit tests for EmbedHandler (highest priority — completely untested)
  8. Integration tests for GPU worker (verify auto-creation fix)
  9. Finalize documentation — fix/complete CONTEXT, Readme, Architecture, create TESTS.md and CLAUDE.md
  10. E2E pipeline test (requires dispatcher service from task 4)

  ---
  Next Step

  Ready to start? Task #1 is the highest-priority blocker: fixing post_repository.py is what enables the CPU worker to run.
