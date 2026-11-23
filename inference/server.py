"""HTTP inference server. Select hardware with INFERENCE_BACKEND=auto|cpu|cuda|neuron."""
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from inference.engine import InferenceEngine

app = FastAPI(title="cuda-inference-service")
engine: Optional[InferenceEngine] = None


class GenerateRequest(BaseModel):
    input_ids: list[int]
    max_new_tokens: int = 32


@app.on_event("startup")
def _startup():
    global engine
    engine = InferenceEngine()


@app.get("/healthz")
def healthz():
    return {"ok": engine is not None, "backend": engine.backend.name if engine else None}


@app.post("/generate")
def generate(req: GenerateRequest):
    return engine.generate(req.input_ids, req.max_new_tokens)
