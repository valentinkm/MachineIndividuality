# registry.py
"""
Model registry for psychometric norm generation.

Each entry defines a model with its adapter configuration,
HuggingFace model identifier, and batch size.
"""

import os
from .adapter import (
    GenericAdapter,
    make_retrying_chat_query,
    mistral_request_kwargs,
    qwen_request_kwargs,
    strip_think_tags,
)

MODEL_REGISTRY = {
    "qwen32b": {
        "adapter": GenericAdapter(
            "Qwen/Qwen3-32B",
            request_kwargs=qwen_request_kwargs,
            clean_text_fn=strip_think_tags,
            cutoff_year=2025,
        ),
        "model_key": "Qwen3-32B",
        "model_name": "Qwen/Qwen3-32B",
        "batch_size": 32,
    },
    "mistral24b": {
        "adapter": GenericAdapter(
            "tgi",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            request_kwargs=mistral_request_kwargs,
            cutoff_year=2023,
        ),
        "model_key": "Mistral-Small-24B-Instruct-2501",
        "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "batch_size": 64,
    },
    "gemma27b": {
        "adapter": GenericAdapter(
            "tgi",
            "google/gemma-3-27b-it",
            "google/gemma-3-27b-it",
            query_fn=make_retrying_chat_query(),
            cutoff_year=2024,
        ),
        "model_key": "gemma-3-27b-it",
        "model_name": "google/gemma-3-27b-it",
        "batch_size": 128,
    },
    "gptoss_20b": {
        "adapter": GenericAdapter(
            "openai/gpt-oss-20b",
            "openai/gpt-oss-20b",
            cutoff_year=2024,
            request_kwargs={
                "reasoning_effort": "low",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        ),
        "model_key": "gpt-oss-20b",
        "model_name": "openai/gpt-oss-20b",
        "batch_size": 32,
    },
    "gptoss_120b": {
        "adapter": GenericAdapter(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-120b",
            request_kwargs={
                "reasoning_effort": "low",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        ),
        "model_key": "gpt-oss-120b",
        "model_name": "openai/gpt-oss-120b",
        "batch_size": 4,
    },
    "olmo3_32b_it": {
        "adapter": GenericAdapter("tgi", "allenai/Olmo-3.1-32B-Instruct"),
        "model_key": "Olmo-3.1-32B-Instruct",
        "model_name": "allenai/Olmo-3.1-32B-Instruct",
        "batch_size": 32,
    },
    "qwen3_235b_it": {
        "adapter": GenericAdapter(
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            request_kwargs=qwen_request_kwargs,
            clean_text_fn=strip_think_tags,
            cutoff_year=2025,
        ),
        "model_key": "Qwen3-235B-A22B-Instruct-2507",
        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "batch_size": 4,
    },
    "falcon_h1_34b_it": {
        "adapter": GenericAdapter("tgi", "tiiuae/Falcon-H1-34B-Instruct"),
        "model_key": "Falcon-H1-34B-Instruct",
        "model_name": "tiiuae/Falcon-H1-34B-Instruct",
        "batch_size": 32,
    },
    "nomos_1": {
        "adapter": GenericAdapter(
            "NousResearch/nomos-1",
            clean_text_fn=strip_think_tags,
        ),
        "model_key": "nomos-1",
        "model_name": "NousResearch/nomos-1",
        "batch_size": 32,
    },
    "granite_4_small": {
        "adapter": GenericAdapter("tgi", "ibm-granite/granite-4.0-h-small"),
        "model_key": "granite-4.0-h-small",
        "model_name": "ibm-granite/granite-4.0-h-small",
        "batch_size": 64,
    },
    "phi_4": {
        "adapter": GenericAdapter("tgi", "microsoft/phi-4"),
        "model_key": "phi-4",
        "model_name": "microsoft/phi-4",
        "batch_size": 64,
    },
    "llama31_8b": {
        "adapter": GenericAdapter("tgi", "meta-llama/Llama-3.1-8B-Instruct"),
        "model_key": "Llama-3.1-8B-Instruct",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 64,
    },
}
