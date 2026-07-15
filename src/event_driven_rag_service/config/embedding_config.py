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

# Keyed by chunk type — determines which model embeds each chunk type.
# Search queries use the config for the chunk type being searched (no separate query entry).
EMBED_CONFIGS: dict[str, EmbedConfig] = {
    "body":          EmbedConfig(model="BAAI/bge-base-en-v1.5",     queue="gpu.embed.bge-base-en-v1.5",     dim=768),
    "title":         EmbedConfig(model="BAAI/bge-small-en-v1.5",    queue="gpu.embed.bge-small-en-v1.5",    dim=384),
    "summary_title": EmbedConfig(model="Qwen3-Embedding-0.6B-Q8_0.gguf", queue="gpu.embed.qwen3-0.6b",      dim=1024),
    "analysis":      EmbedConfig(model="Qwen/Qwen3-0.6B",           queue="gpu.embed.qwen3-0.6b",           dim=1024),
}