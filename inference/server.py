"""HTTP inference server.

Routes: ``/generate`` + ``/healthz`` (service API) and ``/invocations`` + ``/ping``
(SageMaker bring-your-own-container contract; same handler).
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference.engine import InferenceEngine

engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(_app):
    global engine
    engine = InferenceEngine()
    yield


app = FastAPI(title="inference-service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    input_ids: list[int]
    max_new_tokens: int = 32


@app.get("/healthz")
@app.get("/ping")
def healthz():
    return {"ok": engine is not None, "backend": engine.device if engine else None}


@app.post("/generate")
@app.post("/invocations")
def generate(req: GenerateRequest):
    try:
        return engine.generate(req.input_ids, req.max_new_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
