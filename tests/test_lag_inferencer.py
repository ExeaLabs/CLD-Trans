import torch

from modules.lag_inferencer import LaggedEdgeScorer, LearnableLagMatrix


def test_learnable_lag_matrix_bounds() -> None:
    lag = LearnableLagMatrix(num_channels=4, tau_max=0.5)
    tau = lag()
    assert tau.shape == (4, 4)
    assert tau.min() >= 0
    assert tau.max() <= 0.5
    assert torch.allclose(torch.diag(tau), torch.zeros(4))


def test_lagged_edge_scorer_shapes() -> None:
    scorer = LaggedEdgeScorer(num_channels=3, input_dim=8, hidden_dim=16, tau_max=0.2, top_k=1)
    h = torch.randn(2, 3, 12, 8)
    out = scorer(h, sample_rate=32.0)
    assert out.edge_probs.shape == (2, 12, 3, 3)
    assert out.adjacency.shape == out.edge_probs.shape
    assert out.tau.shape == (3, 3)
