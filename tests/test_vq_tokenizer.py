import torch

from modules.vq_tokenizer import PhysiologicalMotifVAE


def test_vq_tokenizer_forward_shapes() -> None:
    model = PhysiologicalMotifVAE(input_channels=1, hidden_dim=16, embed_dim=8, codebook_size=16)
    x = torch.randn(4, 1, 128)
    out = model(x)
    assert out["reconstruction"].shape == x.shape
    assert out["z_q"].shape[-1] == 8
    assert out["indices"].ndim == 2
    loss = out["reconstruction"].pow(2).mean() + out["commit_loss"] + out["codebook_loss"]
    loss.backward()
