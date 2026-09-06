"""Pluggable generation interface used by the notebook and Streamlit website."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_name: str
    input_tokens: int
    output_tokens: int


class LocalQwenProvider:
    """Lazy local Hugging Face provider matching the Configuration A baseline."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Local Qwen dependencies are missing. Install requirements-llm.txt."
            ) from error

        self.model_name = model_name
        self._torch = torch
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.execution_device = self._device.type
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype="auto",
            local_files_only=local_files_only,
        )
        self.model_revision = getattr(self._model.config, "_commit_hash", None) or revision
        self.generation_parameters = {
            "temperature": 0.2,
            "do_sample": True,
            "top_p": 0.9,
            "seed": 42,
            "enable_thinking": False,
            "model_revision": self.model_revision,
            "execution_device": self.execution_device,
        }
        self._model.to(self._device)
        self._model.eval()
        self._torch.manual_seed(42)

    def generate(self, prompt: str, *, max_new_tokens: int = 250) -> GenerationResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a credit-card fraud investigation assistant. Follow the "
                    "experimental condition and output contract in the user request."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.generation_parameters["enable_thinking"],
        )
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        input_tokens = int(inputs["input_ids"].shape[-1])
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.generation_parameters["temperature"],
                do_sample=self.generation_parameters["do_sample"],
                top_p=self.generation_parameters["top_p"],
            )
        generated_tokens = outputs[0, input_tokens:]
        text = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        return GenerationResult(
            text=text,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=int(generated_tokens.shape[-1]),
        )


def provider_is_available() -> bool:
    """Return whether the optional local provider dependencies can be imported."""
    import importlib.util

    return all(
        importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "accelerate")
    )
