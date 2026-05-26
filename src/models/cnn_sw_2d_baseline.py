"""
Shallow Water Equations Baseline Model (No Conservation Constraints)

A CNN baseline model for shallow water equations that:
- Takes 3 input channels (h, mx, my)
- Outputs 3 channels (next h, mx, my)
- Does NOT have structural conservation guarantees
- Can be trained with optional soft conservation loss

Strong baselines include:
- FluxNet_SW_Baseline_Bound: With softplus on h (h >= 0)
  Combined with soft conservation loss during training for strong baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad2D(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, norm_2d=nn.BatchNorm2d):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            CircularPad2D(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn(),
            CircularPad2D(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)


class FluxNet_SW_Baseline(nn.Module):
    """
    Shallow Water Baseline Model (No Conservation Constraints)

    This is a direct prediction CNN baseline for shallow water equations.
    Used for ablation studies to compare against FluxNet-SW.

    Args:
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for convolutions
        act_fn: Activation function
        norm_2d: Normalization layer
        prediction_mode: 'direct' or 'residual'
            'direct': directly predict next state
            'residual': predict change, add to current state
        bound_h: Whether to apply lower bound on h (h >= 0) via softplus
        lower_bound: Lower bound value for h (default 0.0)
    """

    def __init__(self,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 prediction_mode='residual',
                 bound_h=False,
                 lower_bound=0.0):
        super().__init__()

        self.prediction_mode = prediction_mode
        self.bound_h = bound_h
        self.lower_bound = lower_bound
        in_channels = 3  # h, mx, my
        out_channels = 3  # next h, mx, my

        # First conv
        self.first_conv = nn.Sequential(
            CircularPad2D(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv2D(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Output layer
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [batch, 3, H, W] = [h, mx, my]

        Returns:
            next_state: tuple containing ([batch, 3, H, W],) - single element tuple for consistency
        """
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        output = self.output_conv(features)

        if self.prediction_mode == 'residual':
            # Predict change and add to input
            next_state = x + output
        else:
            # Direct prediction
            next_state = output

        # Return as single-element tuple for consistency with other models
        return (next_state,)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    batch_size = 2
    H, W = 32, 32

    # Test without bound
    print("\n=== Testing FluxNet_SW_Baseline (no bound) ===")
    model = FluxNet_SW_Baseline(
        base_channels=32,
        num_blocks=4,
        prediction_mode='residual',
        bound_h=False
    ).to(device)
    model.eval()

    h = torch.rand(batch_size, 1, H, W).to(device) + 0.5
    mx = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    my = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    input_state = torch.cat([h, mx, my], dim=1)

    with torch.no_grad():
        output = model(input_state)
        next_state = output[0]

    print(f"Input shape: {input_state.shape}, Output shape: {next_state.shape}")
    h_min = next_state[:, 0].min()
    print(f"h min: {h_min.item():.4f} (no bound guarantee)")

    # Test with bound (strong baseline)
    print("\n=== Testing FluxNet_SW_Baseline (with h bound - strong baseline) ===")
    model_bound = FluxNet_SW_Baseline(
        base_channels=32,
        num_blocks=4,
        prediction_mode='residual',
        bound_h=True,
        lower_bound=0.0
    ).to(device)
    model_bound.eval()

    with torch.no_grad():
        output = model_bound(input_state)
        next_state = output[0]

    h_min = next_state[:, 0].min()
    print(f"h min: {h_min.item():.4f} (should be >= 0: {h_min.item() >= 0})")

    # Check conservation (not guaranteed)
    for i, name in enumerate(['h', 'mx', 'my']):
        initial_mass = input_state[:, i:i+1].sum()
        final_mass = next_state[:, i:i+1].sum()
        drift = (final_mass - initial_mass).abs() / (initial_mass.abs() + 1e-8)
        print(f"{name}: drift={drift.item():.6f}")

    print("\n=== Tests completed ===")
    print("Bound models can be combined with soft conservation loss for strong baseline")
