"""Compile the decoder for trn1/inf2 with torch_neuronx.trace on a CPU-only host.

``cpu_backend=True`` + ``--target`` runs the full XLA trace and neuronx-cc compile without a
Neuron device, so this proves the model (including the NKI RMSNorm custom call, the explicit
attention, and the bucketed static-shape signature) lowers and compiles to a NEFF. It does not
execute the NEFF: running and timing it requires a trn1/inf2 instance.
"""
import os

import pytest
import torch

torch_neuronx = pytest.importorskip("torch_neuronx")

from inference.kernels.dispatch import KernelConfig  # noqa: E402
from inference.model import load_model  # noqa: E402
from inference.neuron import BucketedDecoder, artifact_path, trace_decoder  # noqa: E402

pytestmark = pytest.mark.neuron_compile


@pytest.fixture(autouse=True)
def _scratch_cwd(tmp_path, monkeypatch):
    # The NKI compile cache and neuronx-cc logs are written to the working directory.
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("target", ["trn1", "inf2"])
def test_trace_decoder_compiles_with_nki_rmsnorm(tiny_cfg, target, tmp_path):
    model = load_model(tiny_cfg, torch.bfloat16, "neuron")
    assert model.kernels == KernelConfig("nki", "reference", "math")
    graph = trace_decoder(model, batch=1, seq=128, target=target, cpu_backend=True,
                          compiler_workdir=str(tmp_path / "wd"))
    assert isinstance(graph, torch.jit.ScriptModule)
    neffs = [f for _, _, files in os.walk(tmp_path / "wd") for f in files if f.endswith(".neff")]
    assert neffs, "neuronx-cc produced no NEFF"


def test_trace_decoder_compiles_with_nki_rope(tiny_cfg, tmp_path):
    model = load_model(tiny_cfg, torch.bfloat16, "neuron", KernelConfig("nki", "nki", "math"))
    trace_decoder(model, batch=1, seq=128, target="trn1", cpu_backend=True,
                  compiler_workdir=str(tmp_path / "wd"))


def test_bucketed_artifacts_are_saved(tiny_cfg, tmp_path):
    # Loading a saved graph (torch.jit.load) initializes the Neuron runtime, which needs a
    # device; that half of the round trip is verified on trn1/inf2 (see MIGRATION.md).
    model = load_model(tiny_cfg, torch.bfloat16, "neuron")
    decoder = BucketedDecoder.compile(model, (128, 256), target="trn1", cpu_backend=True,
                                      artifact_dir=str(tmp_path))
    assert decoder.buckets == (128, 256)
    for seq in (128, 256):
        assert os.path.getsize(artifact_path(str(tmp_path), 1, seq)) > 0
