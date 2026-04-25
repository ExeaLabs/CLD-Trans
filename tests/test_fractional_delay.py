import torch

from modules.fractional_delay import FractionalDelay, delay_signal


def test_fractional_delay_preserves_shape_and_grad() -> None:
    x = torch.randn(2, 32, dtype=torch.double, requires_grad=True)
    tau = torch.tensor(0.03, dtype=torch.double, requires_grad=True)

    def func(signal: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        return delay_signal(signal, lag, sample_rate=128.0, dim=-1)

    assert torch.autograd.gradcheck(func, (x, tau), eps=1e-6, atol=1e-4)


def test_pairwise_delay_shape() -> None:
    x = torch.randn(2, 3, 16, 4)
    tau = torch.zeros(3, 3)
    out = FractionalDelay.apply_channel_pair_delays(x, tau, sample_rate=16.0)
    assert out.shape == (2, 3, 3, 16, 4)
