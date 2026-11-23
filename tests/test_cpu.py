"""CPU tests: device selection, reference path, bucketing, engine, server, benchmark math."""
import pytest
import torch

from inference.device import resolve_device, resolve_dtype
from inference.kernels.dispatch import KernelConfig, apply_rope, rmsnorm
from inference.kernels.reference import apply_rope_reference, rmsnorm_reference
from inference.model import causal_attention_math, load_model, rope_tables
from inference.neuron import BucketedDecoder, pad_to_bucket, parse_buckets, pick_bucket


def test_resolve_device_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    assert resolve_device() == "cpu"
    assert resolve_device("neuron") == "neuron"
    monkeypatch.setenv("INFERENCE_DEVICE", "tpu")
    with pytest.raises(ValueError):
        resolve_device()


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("inference.device.neuron_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr("inference.device.neuron_available", lambda: True)
    assert resolve_device("auto") == "neuron"


def test_cuda_requested_without_gpu_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_dtype_defaults_and_override(monkeypatch):
    assert resolve_dtype("cuda") == torch.float16
    assert resolve_dtype("neuron") == torch.bfloat16
    assert resolve_dtype("cpu") == torch.float32
    monkeypatch.setenv("INFERENCE_DTYPE", "bf16")
    assert resolve_dtype("cpu") == torch.bfloat16


def test_kernel_defaults_per_device(monkeypatch):
    assert KernelConfig.for_device("cuda") == KernelConfig("cuda", "triton", "flash")
    assert KernelConfig.for_device("neuron") == KernelConfig("nki", "reference", "math")
    assert KernelConfig.for_device("cpu") == KernelConfig("reference", "reference", "sdpa")
    monkeypatch.setenv("INFERENCE_ROPE_IMPL", "nki")
    assert KernelConfig.for_device("neuron").rope == "nki"
    with pytest.raises(ValueError):
        KernelConfig(rmsnorm="cudnn")


def test_reference_dispatch_matches_reference():
    x, w = torch.randn(4, 64), torch.randn(64)
    torch.testing.assert_close(rmsnorm(x, w, 1e-6, "reference"), rmsnorm_reference(x, w))
    cos, sin = rope_tables(_cfg(), device="cpu")
    q = torch.randn(1, 2, 16, 32)
    torch.testing.assert_close(apply_rope(q, cos[:16], sin[:16], "reference"),
                               apply_rope_reference(q, cos[:16], sin[:16]))


def _cfg():
    from inference.model import ModelConfig
    return ModelConfig(hidden_size=128, num_heads=4)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_math_attention_matches_sdpa(dtype, tol):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 37, 32, dtype=dtype) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    torch.testing.assert_close(causal_attention_math(q, k, v).float(), ref, atol=tol, rtol=tol)


def test_math_attention_model_matches_sdpa_model(tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 40))
    a = load_model(tiny_cfg, torch.float32, "cpu", KernelConfig(attention="sdpa"))(ids)
    b = load_model(tiny_cfg, torch.float32, "cpu", KernelConfig(attention="math"))(ids)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_buckets():
    assert parse_buckets("512,128,256") == (128, 256, 512)
    assert parse_buckets(None, max_seq_len=256) == (128, 256)
    assert pick_bucket(1, (128, 256)) == 128 and pick_bucket(129, (128, 256)) == 256
    with pytest.raises(ValueError):
        pick_bucket(257, (128, 256))
    padded, last = pad_to_bucket(torch.tensor([[5, 6, 7]]), 8)
    assert padded.tolist() == [[5, 6, 7, 0, 0, 0, 0, 0]] and last.tolist() == [2]


def test_right_padding_does_not_change_next_token_logits(tiny_cfg):
    """The invariant static-shape bucketing on Neuron relies on."""
    model = load_model(tiny_cfg, torch.float32, "cpu", KernelConfig(attention="math"))
    ids = torch.randint(1, tiny_cfg.vocab_size, (1, 21))
    bucketed = BucketedDecoder({32: model, 64: model}, batch_size=1)
    torch.testing.assert_close(bucketed(ids), model(ids)[:, -1, :], atol=1e-5, rtol=1e-5)


def test_bucketed_greedy_generation_matches_eager(tiny_cfg):
    model = load_model(tiny_cfg, torch.float32, "cpu", KernelConfig(attention="math"))
    bucketed = BucketedDecoder({16: model, 32: model}, batch_size=1)
    a = b = torch.tensor([[1, 2, 3]])
    for _ in range(20):  # crosses the 16 -> 32 bucket boundary
        a = torch.cat([a, model(a)[:, -1].argmax(-1, keepdim=True)], 1)
        b = torch.cat([b, bucketed(b).argmax(-1, keepdim=True)], 1)
    assert a.tolist() == b.tolist()


def test_generate_on_cpu(tiny_cfg):
    from inference.engine import InferenceEngine
    eng = InferenceEngine(tiny_cfg, device="cpu")
    out = eng.generate([1, 2, 3], max_new_tokens=4)
    assert len(out["output_ids"]) == 7 and out["backend"] == "cpu" and out["device_ms"] > 0
    with pytest.raises(ValueError):
        eng.generate([1] * tiny_cfg.max_seq_len, max_new_tokens=1)


def test_server_routes(monkeypatch, tiny_cfg):
    from fastapi.testclient import TestClient

    import inference.server as server
    from inference.engine import InferenceEngine

    monkeypatch.setattr(server, "InferenceEngine", lambda: InferenceEngine(tiny_cfg, device="cpu"))
    with TestClient(server.app) as client:
        assert client.get("/ping").json() == {"ok": True, "backend": "cpu"}
        for route in ("/generate", "/invocations"):
            r = client.post(route, json={"input_ids": [1, 2, 3], "max_new_tokens": 2})
            assert r.status_code == 200 and len(r.json()["output_ids"]) == 5
        assert client.post("/generate", json={"input_ids": [], "max_new_tokens": 2}).status_code == 400


def test_cost_per_million_tokens():
    from benchmarks.bench_latency import cost_per_million_tokens, resolve_price
    # $3.60/hr at 1000 tok/s on one replica -> $0.001/s / 1000 tok/s * 1e6 = $1.00 per 1M tokens
    assert cost_per_million_tokens(3.60, 1000.0) == pytest.approx(1.0)
    assert cost_per_million_tokens(3.60, 1000.0, replicas=2) == pytest.approx(0.5)
    price, replicas, _ = resolve_price("inf2.xlarge", None)
    assert price > 0 and replicas == 2
    assert resolve_price("inf2.xlarge", 9.99)[0] == 9.99
    assert resolve_price(None, None)[0] is None


def test_benchmark_runs_on_cpu(capsys):
    import json

    from benchmarks.bench_latency import main
    main(["--device", "cpu", "--model", "tiny", "--batch", "2", "--seq", "32", "--iters", "2",
          "--warmup", "1", "--instance", "c7i.xlarge"])
    r = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert r["correctness"]["top1_agreement"] == 1.0
    assert r["tokens_per_s"] > 0 and r["usd_per_1m_tokens"] > 0


def test_benchmark_seq_defaults_to_model_limit(capsys):
    import json

    from benchmarks.bench_latency import main
    main(["--device", "cpu", "--model", "tiny", "--batch", "1", "--iters", "1", "--warmup", "0"])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["seq"] == 256
    with pytest.raises(SystemExit):
        main(["--device", "cpu", "--model", "tiny", "--seq", "512"])
