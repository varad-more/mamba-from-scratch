from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mamba_minimal import api as api_module
from mamba_minimal.generate import LoadedGenerationModel


class DummyModel:
    pass


class DummyTokenizer:
    pass


def test_healthz_returns_ok() -> None:
    app = api_module.create_app()
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_generate_uses_generation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_generation_model(model_name: str, device: str):
        return LoadedGenerationModel(
            model_name=model_name,
            device="cpu" if device == "auto" else device,
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
        )

    def fake_generate_with_loaded_model(loaded, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float):
        assert loaded.model_name == "demo-model"
        assert prompt == "hello"
        assert max_new_tokens == 4
        assert do_sample is False
        assert temperature == 1.0
        return "hello world"

    monkeypatch.setattr(api_module, "load_generation_model", fake_load_generation_model)
    monkeypatch.setattr(api_module, "generate_with_loaded_model", fake_generate_with_loaded_model)

    app = api_module.create_app()
    client = TestClient(app)
    response = client.post(
        "/generate",
        json={
            "prompt": "hello",
            "model_name": "demo-model",
            "device": "auto",
            "max_new_tokens": 4,
            "do_sample": False,
            "temperature": 1.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "demo-model"
    assert payload["output_text"] == "hello world"
    assert payload["device"] == "cpu"
