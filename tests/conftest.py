import pytest

from inference.model import ModelConfig


@pytest.fixture
def tiny_cfg():
    return ModelConfig(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4,
                       intermediate_size=256, max_seq_len=256)
