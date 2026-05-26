"""
FNO for Shallow Water Equations

Fourier Neural Operator for predicting shallow water dynamics.
Input: [h, mx, my] at time t
Output: [h, mx, my] at time t+1

No structural conservation guarantees - serves as a baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer (Fourier Layer)"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in first dimension
        self.modes2 = modes2  # Number of Fourier modes in second dimension

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Handle positive frequencies in first dimension
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Handle negative frequencies in first dimension
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Compute inverse 2D FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x


class FNOBlock(nn.Module):
    """Single FNO Block: Spectral Conv + Local Conv + Activation"""

    def __init__(self, channels, modes1, modes2, act_fn=nn.GELU):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = act_fn()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        return self.act(x1 + x2)


class FNO_SW(nn.Module):
    """
    Fourier Neural Operator for Shallow Water Equations

    Args:
        modes1: Number of Fourier modes in first spatial dimension
        modes2: Number of Fourier modes in second spatial dimension
        width: Width of hidden channels
        num_layers: Number of FNO blocks
        in_channels: Number of input channels (default 3: h, mx, my)
        out_channels: Number of output channels (default 3: h, mx, my)
        prediction_mode: 'direct' or 'residual'
    """

    def __init__(self,
                 modes1=16,
                 modes2=16,
                 width=64,
                 num_layers=4,
                 in_channels=3,
                 out_channels=3,
                 prediction_mode='residual'):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.prediction_mode = prediction_mode

        # Lift to higher dimension
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2)
            for _ in range(num_layers)
        ])

        # Project back to output dimension
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: [batch, 3, H, W]
        """
        # Lift
        x_lifted = self.lift(x)

        # FNO blocks
        for block in self.fno_blocks:
            x_lifted = block(x_lifted)

        # Project
        output = self.project(x_lifted)

        # Apply prediction mode
        if self.prediction_mode == 'residual':
            next_state = x + output
        else:
            next_state = output

        return (next_state,)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing FNO_SW on: {device}")

    batch_size = 4
    H, W = 64, 64

    model = FNO_SW(
        modes1=16,
        modes2=16,
        width=32,
        num_layers=4,
        prediction_mode='residual'
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create input
    h = torch.rand(batch_size, 1, H, W).to(device) + 0.5
    mx = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    my = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    input_state = torch.cat([h, mx, my], dim=1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output, = model(input_state)

    print(f"Input shape: {input_state.shape}")
    print(f"Output shape: {output.shape}")

    # Conservation check (FNO doesn't guarantee conservation)
    for i, name in enumerate(['h', 'mx', 'my']):
        initial_mass = input_state[:, i].sum()
        final_mass = output[:, i].sum()
        drift = (final_mass - initial_mass).abs() / (initial_mass.abs() + 1e-8)
        print(f"{name} drift: {drift.item():.4e} (FNO has no conservation guarantee)")
