from __future__ import annotations

import torch

from mamba_minimal.generate import LoadedGenerationModel, generate_with_loaded_model, resolve_device


class FakeBatch(dict):
    def to(self, device: str):
        self["device"] = device
        return self


class FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]]), "prompt": prompt})

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return f"decoded:{list(token_ids)}"


class FakeModel:
    def generate(self, **kwargs):
        assert "input_ids" in kwargs
        return torch.tensor([[4, 5, 6]])


def test_resolve_device_auto_returns_cpu_or_cuda() -> None:
    resolved = resolve_device("auto")
    assert resolved in {"cpu", "cuda"}


def test_generate_with_loaded_model_uses_loaded_objects() -> None:
    loaded = LoadedGenerationModel(
        model_name="fake-model",
        device="cpu",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )

    text = generate_with_loaded_model(loaded, prompt="hello", max_new_tokens=4)
    assert text == "decoded:[tensor(4), tensor(5), tensor(6)]"
