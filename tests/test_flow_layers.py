import torch

from modules.flow_layers import CLDOdeBlock


def test_ode_block_forward_and_grad() -> None:
    block = CLDOdeBlock(hidden_dim=8, solver="rk4")
    h0 = torch.randn(2, 3, 8, requires_grad=True)
    t = torch.linspace(0, 1, 5)
    adjacency = torch.rand(2, 5, 3, 3)
    out = block(h0, t, adjacency)
    assert out.trajectory.shape == (2, 5, 3, 8)
    loss = out.trajectory.pow(2).mean()
    loss.backward()
    assert h0.grad is not None
