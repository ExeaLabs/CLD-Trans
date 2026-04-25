import torch

from models.cld_transformer import CLDTransformer, CLDTransformerConfig


def test_cld_transformer_end_to_end_smoke() -> None:
    cfg = CLDTransformerConfig(
        num_channels=4,
        num_classes=3,
        codebook_size=16,
        motif_dim=8,
        hidden_dim=8,
        sample_rate=64.0,
        tau_max=0.25,
    )
    model = CLDTransformer(cfg)
    x = torch.randn(2, 4, 128)
    out = model(x, mode="fine_tune")
    assert out["logits"].shape == (2, 3)
    assert out["tau"].shape == (4, 4)
    loss = out["logits"].pow(2).mean() + out["commit_loss"] + out["codebook_loss"]
    loss.backward()
