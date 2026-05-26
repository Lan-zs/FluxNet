"""
CNN Baseline Models for Ablation Studies

These models directly predict the next state without structural conservation guarantees.
Used as baselines to compare against FluxNet models.

Strong baselines include:
- Bounded output (softplus for lower bound, sigmoid for double bound)
- Combined with soft conservation loss during training

Bound modes:
- 'none': No bound constraint
- 'lower': Lower bound via softplus (useful for c>=0 or h>=0)
- 'double': Double bound via sigmoid (useful for 0<=rho<=1 or 0<=phi<=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1D components
class CircularPad1D(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding), mode='circular')


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, norm_1d=nn.BatchNorm1d):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            CircularPad1D(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            norm_1d(out_channels),
            act_fn(),
            CircularPad1D(padding),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            norm_1d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)


# 2D components
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


class CNN_Baseline_1D(nn.Module):
    """
    1D CNN Baseline for direct prediction

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (usually 1)
        base_channels: Base feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolutional kernel size
        prediction_mode: 'direct' or 'residual'
        bound_mode: 'none', 'lower', or 'double'
            - 'none': No bound constraint
            - 'lower': Output >= lower_bound via softplus
            - 'double': lower_bound <= output <= upper_bound via sigmoid
        lower_bound: Lower bound value (default 0.0)
        upper_bound: Upper bound value (default 1.0)
    """

    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_1d=nn.BatchNorm1d,
                 prediction_mode='residual',
                 bound_mode='none',
                 lower_bound=0.0,
                 upper_bound=1.0):
        super().__init__()

        self.prediction_mode = prediction_mode
        self.bound_mode = bound_mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # First conv
        self.first_conv = nn.Sequential(
            CircularPad1D(kernel_size // 2),
            nn.Conv1d(in_channels, base_channels, kernel_size, padding=0),
            norm_1d(base_channels),
            act_fn()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv1D(base_channels, base_channels, kernel_size, act_fn, norm_1d),
                nn.Conv1d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Output layer
        self.output_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, length]

        Returns:
            next_field: [batch, out_channels, length]
        """
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        output = self.output_conv(features)

        if self.prediction_mode == 'residual':
            next_field = x[:, 0:1] + output
        else:
            next_field = output

        # Apply bound constraints
        if self.bound_mode == 'lower':
            # Use softplus to ensure >= lower_bound
            next_field = F.softplus(next_field - self.lower_bound) + self.lower_bound
        elif self.bound_mode == 'double':
            # Use sigmoid to map to [lower_bound, upper_bound]
            range_size = self.upper_bound - self.lower_bound
            next_field = torch.sigmoid(next_field) * range_size + self.lower_bound

        return next_field,


class CNN_Baseline_2D(nn.Module):
    """
    2D CNN Baseline for direct prediction

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolutional kernel size
        prediction_mode: 'direct' or 'residual'
        bound_mode: 'none', 'lower', or 'double'
        lower_bound: Lower bound value
        upper_bound: Upper bound value
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 prediction_mode='residual',
                 bound_mode='none',
                 lower_bound=0.0,
                 upper_bound=1.0):
        super().__init__()

        self.prediction_mode = prediction_mode
        self.bound_mode = bound_mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

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
            x: [batch, in_channels, H, W]

        Returns:
            next_field: [batch, out_channels, H, W]
        """
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        output = self.output_conv(features)

        if self.prediction_mode == 'residual':
            next_field = x[:, 0:1] + output
        else:
            next_field = output

        # Apply bound constraints
        if self.bound_mode == 'lower':
            next_field = F.softplus(next_field - self.lower_bound) + self.lower_bound
        elif self.bound_mode == 'double':
            range_size = self.upper_bound - self.lower_bound
            next_field = torch.sigmoid(next_field) * range_size + self.lower_bound

        return next_field,


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    # Test 1D baseline
    print("\n=== Testing 1D CNN Baseline (no bound) ===")
    model_1d = CNN_Baseline_1D(in_channels=2, out_channels=1).to(device)
    x_1d = torch.randn(4, 2, 64).to(device)
    out_1d, = model_1d(x_1d)
    print(f"Input shape: {x_1d.shape}, Output shape: {out_1d.shape}")
    print(f"Output range: [{out_1d.min().item():.3f}, {out_1d.max().item():.3f}]")

    # Test 1D with lower bound
    print("\n=== Testing 1D CNN Baseline (lower bound) ===")
    model_1d_lower = CNN_Baseline_1D(in_channels=2, out_channels=1, bound_mode='lower', lower_bound=0.0).to(device)
    out_1d_lower, = model_1d_lower(x_1d)
    print(f"Output range: [{out_1d_lower.min().item():.3f}, {out_1d_lower.max().item():.3f}]")
    print(f"Min >= 0: {out_1d_lower.min().item() >= 0}")

    # Test 1D with double bound
    print("\n=== Testing 1D CNN Baseline (double bound) ===")
    model_1d_double = CNN_Baseline_1D(in_channels=2, out_channels=1, bound_mode='double').to(device)
    out_1d_double, = model_1d_double(x_1d)
    print(f"Output range: [{out_1d_double.min().item():.3f}, {out_1d_double.max().item():.3f}]")
    print(f"In [0,1]: {out_1d_double.min().item() >= 0 and out_1d_double.max().item() <= 1}")

    # Test 2D baseline
    print("\n=== Testing 2D CNN Baseline ===")
    model_2d = CNN_Baseline_2D(in_channels=1, out_channels=1).to(device)
    x_2d = torch.randn(4, 1, 32, 32).to(device)
    out_2d, = model_2d(x_2d)
    print(f"Input shape: {x_2d.shape}, Output shape: {out_2d.shape}")

    # Test 2D with double bound (for spinodal decomposition)
    print("\n=== Testing 2D CNN Baseline (double bound for spinodal) ===")
    model_2d_double = CNN_Baseline_2D(in_channels=1, bound_mode='double').to(device)
    out_2d_double, = model_2d_double(x_2d)
    print(f"Output range: [{out_2d_double.min().item():.3f}, {out_2d_double.max().item():.3f}]")

    print("\nBaseline models OK!")
