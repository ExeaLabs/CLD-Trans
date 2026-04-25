from analysis.identifiability import edge_support_f1, tau_mae
from data.synthetic_ldsem import generate_ldsem_batch


def test_synthetic_ldsem_has_ground_truth() -> None:
    batch = generate_ldsem_batch(batch_size=4, num_channels=4, num_steps=64, seed=123)
    assert batch.x.shape == (4, 4, 64)
    assert batch.tau.shape == (4, 4)
    assert batch.graph.sum() > 0
    assert tau_mae(batch.tau, batch.tau, batch.graph) == 0.0
    probs = batch.graph[None, None].expand(2, 3, 4, 4)
    assert edge_support_f1(probs, batch.graph) == 1.0
