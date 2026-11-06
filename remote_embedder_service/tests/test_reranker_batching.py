"""_RerankerEncoder scores a candidate pool in small sub-batches.

A 99-document pool in one forward pass (full-vocab logits for every position)
overflowed an 8 GB GPU and ran for minutes; the encoder now batches and asks the
model for only the last position's logits. Fake model/tokenizer, no GPU.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from remote_embedder import manager
from remote_embedder.manager import _RerankerEncoder


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]

    def encode(self, text, add_special_tokens=False):
        return [9, 9]

    def __call__(self, pairs, **kwargs):
        # One token per pair: the number after "<Document>: ", so the fake model can score it.
        return {"input_ids": [[int(p.split("<Document>: ")[1])] for p in pairs]}

    def pad(self, inputs, padding=True, return_tensors="pt"):
        return {"input_ids": torch.tensor(inputs["input_ids"])}


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.logits_to_keep_args: list[object] = []

    def __call__(self, input_ids, logits_to_keep=None):
        self.batch_sizes.append(input_ids.shape[0])
        self.logits_to_keep_args.append(logits_to_keep)
        logits = torch.zeros(input_ids.shape[0], 1, 2)
        logits[:, 0, 1] = input_ids[:, 2].float()  # "yes" logit = the document's number
        return SimpleNamespace(logits=logits)


def _encoder():
    model = _FakeModel()
    return _RerankerEncoder(model, _FakeTokenizer()), model


def test_pool_is_scored_in_sub_batches_and_keeps_input_order():
    encoder, model = _encoder()
    docs = [str(i) for i in range(20)]

    scores = encoder.score("q", docs)

    assert model.batch_sizes == [8, 8, 4]
    assert len(scores) == 20
    assert scores == sorted(scores)  # score rises with the doc number, so order was preserved


def test_only_the_last_position_logits_are_requested():
    encoder, model = _encoder()
    encoder.score("q", ["1", "2"])
    assert model.logits_to_keep_args == [1]


def test_batch_size_is_small_enough_for_an_8gb_card():
    assert manager._RERANK_BATCH_SIZE <= 8
