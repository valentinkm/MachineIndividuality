# backend.py
"""
Offline vLLM backend for direct GPU inference.
"""

import os
from typing import List, Dict, Any

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


class OfflineVLLMBackend:
    """
    Backend for running vLLM directly in-process (offline inference).
    Used for SLURM batch jobs with direct GPU access.
    """

    name = "offline_vllm"

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
    ):
        if LLM is None:
            raise ImportError(
                "vLLM not found. Install with `pip install vllm` to use offline backend."
            )
        print(f"Initializing Offline vLLM: {model_path} (TP={tensor_parallel_size})")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

    def chat_batch(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Generate completions for a batch of chat conversations.
        Each conversation is a list of ``{"role": "...", "content": "..."}`` dicts.
        """
        if not conversations:
            return []

        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 256)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        stop = kwargs.get("stop", None)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

        print(f"Offline Chat: {len(conversations)} conversations | Temp: {temperature}")

        chat_template_kwargs = kwargs.get(
            "chat_template_kwargs", {"enable_thinking": False}
        )

        outputs = self.llm.chat(
            conversations,
            params,
            use_tqdm=True,
            chat_template_kwargs=chat_template_kwargs,
        )

        return [output.outputs[0].text for output in outputs]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate completions for raw prompts (base models without chat template)."""
        if not prompts:
            return []

        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 256)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

        print(f"Offline Gen: {len(prompts)} prompts | Temp: {temperature}")

        outputs = self.llm.generate(prompts, params, use_tqdm=True)
        return [output.outputs[0].text for output in outputs]
