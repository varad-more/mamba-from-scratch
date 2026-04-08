from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from time import perf_counter

import torch

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - optional dependency
    FastAPI = None
    HTTPException = RuntimeError
    BaseModel = object
    Field = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from .generate import DEFAULT_MODEL_NAME, generate_with_loaded_model, load_generation_model, resolve_device


@dataclass(slots=True)
class HealthResponse:
    status: str
    device: str
    cuda_available: bool


if BaseModel is not object:

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1)
        model_name: str = Field(default=DEFAULT_MODEL_NAME)
        device: str = Field(default="auto")
        max_new_tokens: int = Field(default=64, ge=1, le=512)
        do_sample: bool = Field(default=False)
        temperature: float = Field(default=1.0, gt=0.0, le=5.0)


    class GenerateResponse(BaseModel):
        model_name: str
        device: str
        output_text: str
        latency_ms: float
        prompt_chars: int
else:
    GenerateRequest = object
    GenerateResponse = object


def create_app() -> FastAPI:
    if FastAPI is None:
        raise RuntimeError(f"fastapi is required for the API server: {IMPORT_ERROR}")

    app = FastAPI(title="Mamba From Scratch API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        payload = HealthResponse(
            status="ok",
            device=resolve_device("auto"),
            cuda_available=torch.cuda.is_available(),
        )
        return asdict(payload)

    @app.post("/generate")
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            loaded = load_generation_model(
                model_name=request.model_name,
                device=request.device,
            )
            start = perf_counter()
            output_text = generate_with_loaded_model(
                loaded,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
            )
            latency_ms = (perf_counter() - start) * 1000.0
        except Exception as exc:  # pragma: no cover - surface runtime errors cleanly
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return GenerateResponse(
            model_name=loaded.model_name,
            device=loaded.device,
            output_text=output_text,
            latency_ms=latency_ms,
            prompt_chars=len(request.prompt),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Mamba-from-scratch generation API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"uvicorn is required to run the API server: {exc}") from exc

    uvicorn.run("mamba_minimal.api:create_app", host=args.host, port=args.port, factory=True)


if __name__ == "__main__":
    main()
