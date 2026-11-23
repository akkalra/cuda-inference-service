import pytest
import torch

from inference.backends import CPUBackend
from inference.engine import InferenceEngine, static_length


class StaticCPUBackend(CPUBackend):
    """CPU with the Neuron generate loop (fixed-length padded inputs)."""

    requires_static_shapes = True


def test_generate_cpu(tiny_cfg):
    out = InferenceEngine(tiny_cfg, backend="cpu").generate([1, 2, 3], max_new_tokens=4)
    assert len(out["output_ids"]) == 7 and out["output_ids"][:3] == [1, 2, 3]
    assert out["backend"] == "cpu" and out["device"] == "cpu"
    assert out["device_ms"] > 0 and out["wall_ms"] > 0
    assert "gpu_ms" not in out


def test_generate_is_deterministic(tiny_cfg):
    eng = InferenceEngine(tiny_cfg, backend="cpu")
    assert eng.generate([5, 6], 6)["output_ids"] == eng.generate([5, 6], 6)["output_ids"]


@pytest.mark.parametrize("prompt,new", [([1, 2, 3], 8), (list(range(1, 60)), 10)])
def test_static_shape_loop_matches_dynamic_loop(tiny_cfg, prompt, new):
    dyn = InferenceEngine(tiny_cfg, backend="cpu").generate(prompt, new)["output_ids"]
    static = InferenceEngine(tiny_cfg, backend=StaticCPUBackend()).generate(prompt, new)["output_ids"]
    assert static == dyn


def test_static_loop_uses_one_input_shape(tiny_cfg):
    eng = InferenceEngine(tiny_cfg, backend=StaticCPUBackend())
    shapes = set()
    model = eng.forward

    def spy(ids, last_pos=None):
        shapes.add((tuple(ids.shape), None if last_pos is None else tuple(last_pos.shape)))
        return model(ids, last_pos)

    eng.forward = spy
    eng.generate([1, 2, 3], 12)
    assert shapes == {((1, 64), (1,))}


def test_static_length(tiny_cfg):
    assert static_length(tiny_cfg, 1) == 64
    assert static_length(tiny_cfg, 65) == 128
    assert static_length(tiny_cfg, 256) == 256
    with pytest.raises(ValueError, match="max_seq_len"):
        static_length(tiny_cfg, 257)


def test_server_routes(monkeypatch, tiny_cfg):
    from fastapi.testclient import TestClient

    import inference.server as server

    monkeypatch.setenv("INFERENCE_BACKEND", "cpu")
    monkeypatch.setattr(server, "InferenceEngine", lambda: InferenceEngine(tiny_cfg))
    with TestClient(server.app) as client:
        assert client.get("/healthz").json() == {"ok": True, "backend": "cpu"}
        r = client.post("/generate", json={"input_ids": [1, 2, 3], "max_new_tokens": 2})
        assert r.status_code == 200 and len(r.json()["output_ids"]) == 5
