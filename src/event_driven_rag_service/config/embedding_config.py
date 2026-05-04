from dataclasses import dataclass


@dataclass
class ChunkConfig:
    strategy: str
    target_words: int       # desired chunk size in words
    chunk_overlap: float    # fraction of previous chunk to prepend as overlap (e.g. 0.10)


@dataclass
class EmbedConfig:
    model: str    # determines the GPU queue to route to
    queue: str    # RabbitMQ routing key: "gpu.embed.{model}"
    dim: int      # embedding vector dimension (required for pgvector column declaration)


CHUNK_CONFIG = ChunkConfig(
    strategy="boundary",
    target_words=500,
    chunk_overlap=0.10,
)

# Keyed by task_type — determines which model embeds each chunk type.
# "summary_title" is treated the same as "summary" for embedding purposes.
EMBED_CONFIGS: dict[str, EmbedConfig] = {
    "body":          EmbedConfig(model="bge-base-v1.5", queue="gpu.embed.bge-base-v1.5", dim=768),
    "summary_title": EmbedConfig(model="bge-base-v1.5", queue="gpu.embed.bge-base-v1.5", dim=768),
    "query":         EmbedConfig(model="bge-base-v1.5", queue="gpu.embed.bge-base-v1.5", dim=768),
    "analysis":      EmbedConfig(model="qwen3-0.6b",    queue="gpu.embed.qwen3-0.6b",    dim=1024),
}