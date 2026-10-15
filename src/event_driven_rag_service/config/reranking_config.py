from dataclasses import dataclass


@dataclass
class RerankConfig:
    # Canonical identity of the model, in HF `org/name` form — mirrors
    # EmbedConfig.primary. Selects the GPU queue and rides on
    # SearchJobCreatedEvent.rerank_profile / SearchRankTask.rerank_profile.
    primary: str
    queue: str    # RabbitMQ routing key: "gpu.rerank.{model}"

    # --- where the work may happen -----------------------------------------
    # Same semantics as EmbedConfig: both legs opt-in independently.
    # allow_local=False means this box genuinely cannot run the model — a
    # failed remote batch should fall back to ANN-only results (via
    # SearchFinalizeTask), not silently load several GB onto a CPU.
    allow_local: bool = True
    allow_remote: bool = True

    # Model id to send to the remote reranker service, when it differs from
    # `primary`. None means "same as primary".
    remote_model: str | None = None

    # Requested weight precision, e.g. "int4", "int8". Applied the same way
    # both legs: bitsandbytes at load time (load_in_4bit / load_in_8bit).
    # None means full precision.
    quantization: str | None = None

    # Per-model override for GpuEmbedWorker's messages-per-poll. Rerank scores
    # (query, document) pairs one candidate at a time under the hood, so this
    # bounds how many candidate chunks are fetched/scored per SearchRankTask
    # batch pulled off the queue, not the number of pairs per forward pass.
    max_batch: int | None = None

    @property
    def remote_id(self) -> str:
        """The id to send to the remote service. Only meaningful when
        `allow_remote` — callers should check that first."""
        return self.remote_model or self.primary


# Keyed by `rerank_profile` — the string carried on SearchJobCreatedEvent and
# SearchRankTask, analogous to EMBED_CONFIGS' chunk-type keys. "default" is
# what the API uses when a caller asks for rerank without naming a profile.
#
# Qwen3-Reranker-4B is the intended production model. Qwen3-Reranker-0.6B is
# kept alongside it for fast local dev and the live-gated integration test —
# small enough to load on a modest GPU (or even CPU) in seconds, so the model
# plumbing can be exercised without waiting on the 4B download/load.
qwen3_reranker_4b_cfg = RerankConfig(
    primary="Qwen/Qwen3-Reranker-4B",
    queue="gpu.rerank.qwen3-reranker-4b",
    remote_model="text-rerank-qwen3-reranker-4b",
    # int4 (nf4) — a deliberately more aggressive default than the int8 used
    # for embedding models: a reranker only needs to preserve *relative*
    # ordering of scores within one candidate pool, not absolute cosine
    # geometry, so it tolerates lower precision better than an embedding space
    # that gets compared across pool boundaries via ANN.
    quantization="int4",
    max_batch=50,
)

qwen3_reranker_0_6b_cfg = RerankConfig(
    primary="Qwen/Qwen3-Reranker-0.6B",
    queue="gpu.rerank.qwen3-reranker-0.6b",
    remote_model="text-rerank-qwen3-reranker-0.6b",
    quantization="int4",
    max_batch=50,
)

RERANK_CONFIGS: dict[str, RerankConfig] = {
    "default": qwen3_reranker_4b_cfg,
    "qwen3-reranker-4b": qwen3_reranker_4b_cfg,
    "qwen3-reranker-0.6b": qwen3_reranker_0_6b_cfg,
}
