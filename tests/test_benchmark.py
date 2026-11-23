import json

import pytest

from benchmarks import bench_latency


def test_benchmark_cpu_end_to_end(tmp_path):
    out = tmp_path / "r.json"
    r = bench_latency.main(["--backend", "cpu", "--preset", "small", "--batch", "2", "--seq", "16",
                            "--warmup", "1", "--iters", "3", "--json-out", str(out)])
    assert json.loads(out.read_text()) == r
    assert r["backend"] == "cpu" and r["batch"] == 2 and r["seq"] == 16 and not r["compiled"]
    assert r["setup_s"] > 0 and r["compile_s"] > 0 and len(r["warmup_ms"]) == 1
    lat = r["latency_ms"]
    assert 0 < lat["p50"] <= lat["p90"] <= lat["p99"]
    assert r["throughput"]["requests_per_s"] == pytest.approx(2 / (lat["mean"] / 1000), rel=0.01)
    acc = r["accuracy_vs_fp32_cpu"]
    assert acc["max_abs_err"] == 0 and acc["top1_agreement"] == 1.0


def test_benchmark_bf16_accuracy_is_reported():
    r = bench_latency.main(["--backend", "cpu", "--preset", "small", "--dtype", "bfloat16",
                            "--batch", "1", "--seq", "8", "--warmup", "0", "--iters", "1"])
    assert r["dtype"] == "bfloat16"
    assert r["accuracy_vs_fp32_cpu"]["cosine_similarity"] > 0.99


@pytest.mark.parametrize("bad", [["--seq", "0"], ["--seq", "4096"], ["--batch", "0"], ["--iters", "0"]])
def test_benchmark_rejects_bad_args(bad):
    with pytest.raises(SystemExit):
        bench_latency.parse_args(bad)


def test_percentile():
    assert bench_latency._percentile([1, 2, 3, 4], 0.5) == 2.5
    assert bench_latency._percentile([5], 0.99) == 5


def test_benchmark_rejects_float32_with_flash_attention(monkeypatch):
    from inference.backends import CPUBackend, KernelConfig

    class FlashBackend(CPUBackend):
        kernels = KernelConfig(rmsnorm="reference", rope="reference", attention="flash")

    monkeypatch.setattr(bench_latency, "get_backend", lambda name=None: FlashBackend())
    with pytest.raises(ValueError, match="float32"):
        bench_latency.main(["--preset", "small", "--dtype", "float32", "--batch", "1", "--seq", "8"])
