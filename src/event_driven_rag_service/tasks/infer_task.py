"""
Infer task: run an LLM inference call against a post (local model or remote API).

task_type variants
------------------
categorise — classify the post into a category using a prompt template
summarise  — generate or improve a summary using a prompt template

The ``prompt_template`` field carries a named template identifier (e.g.
``"analyse_post_v1"``).  Workers resolve the template by name rather than
receiving prompt text inline, keeping message sizes small and decoupling
prompt management from task dispatch.
"""
from __future__ import annotations

from typing import Literal

from .base_task import BaseTask


class InferTask(BaseTask):
    kind: Literal["infer"] = "infer"
    task_type: Literal["categorise", "summarise"] = "categorise"
    post_id: int
    post_table: str
    model: str              # "qwen3.5-4b" | "chatgpt-4o"
    prompt_template: str    # named template; worker resolves text from a registry
