from typing import List

from .base_event import BaseEvent


# class EmbeddingBatchRequestedEvent(BaseEvent):
#     """
#     Internal orchestration event for batching.

#     Emitted by dispatcher or batcher.
#     """

#     event_type: str = "embedding.batch.requested"

#     post_id: int
#     post_table: str
#     chunk_ids: List[str]
#     chunk_table: str
#     model_name: str



class EmbeddingCompletedEvent(BaseEvent):
    """
    Fired when embeddings are successfully computed.
    """

    event_type: str = "embedding.completed"

    post_id: int
    post_table: str
    chunk_ids: List[str]
    chunk_table: str
    model_name: str