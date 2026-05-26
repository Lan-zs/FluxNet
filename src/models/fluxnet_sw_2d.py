"""
FluxNet-SW 2D: Shallow Water Equations Model

Head configurations:
- NNN: No constraint on all three fields (h, mx, my)
- LNN: Lower bound on h (h >= 0), no constraint on mx, my
- LPP: Lower bound on h, Positive fluxes for mx, my
- LPP_h: Lower bound on h, Positive fluxes for mx, my scaled by h
         (momentum flux = h * potential, ensures no momentum at dry regions)

Optimization: Merge the roll loops for all three fields into a single loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad(nn.Module):
    def __init__(self, padding):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, norm_2d=nn.BatchNorm2d):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            CircularPad(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn(),
            CircularPad(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)


class FluxNet_SW_2D(nn.Module):
    """
    Shallow Water Equations Multi-Head Model

    Args:
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolution kernel size
        act_fn: Activation function
        norm_2d: Normalization layer
        neighborhood_size: Size of flux neighborhood (must be odd)
        lower_bound: Lower bound for h field (default 0.0)
        head_config: Head configuration string
            - 'NNN': No constraint on all fields (original UUU)
            - 'LNN': Lower bound on h, no constraint on mx, my (original LUU)
            - 'LPP': Lower bound on h, positive fluxes for mx, my
            - 'LPP_h': Lower bound on h, momentum flux = h * potential (best for dry regions)
    """

    def __init__(self,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 neighborhood_size=15,
                 lower_bound=0.0,
                 head_config='LNN'):
        super().__init__()

        self.head_config = head_config
        self.register_buffer('lower_bound', torch.tensor(lower_bound))

        self.num_neighbors = neighborhood_size * neighborhood_size - 1
        self.neighborhood_size = neighborhood_size

        # Shared feature extractor
        self.first_conv = nn.Sequential(
            CircularPad(kernel_size // 2),
            nn.Conv2d(3, base_channels, kernel_size=kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Head-specific flux convolutions
        if head_config in ['LPP']:
            # h uses L-head: outflow percentage + distribution ratios
            self.h_flux_conv = nn.Conv2d(base_channels, 1 + self.num_neighbors, kernel_size=1)
        else:  # NNN
            self.h_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)

        # mx, my heads
        self.mx_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)
        self.my_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)

        # Pre-compute neighbor offsets
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i != 0 or j != 0:
                    neighbor_offsets.append((i, j))
        self.neighbor_offsets_list = neighbor_offsets
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: [batch, 3, H, W]
            h_delta: Change in h
            mx_delta: Change in mx
            my_delta: Change in my
        """
        # Feature extraction
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        h = x[:, 0:1]
        mx = x[:, 1:2]
        my = x[:, 2:3]

        # Compute based on head configuration
        if self.head_config == 'NNN':
            next_h, next_mx, next_my = self._compute_all_NNN(h, mx, my, features)
        elif self.head_config == 'LPP':
            next_h, next_mx, next_my = self._compute_all_LPP(h, mx, my, features)

        else:
            raise ValueError(f"Unknown head_config: {self.head_config}")

        next_state = torch.cat([next_h, next_mx, next_my], dim=1)

        return (next_state,
                next_h - h,
                next_mx - mx,
                next_my - my)

    def _compute_all_PPP(self, h, mx, my, features):
        """
        NNN mode: No constraint on all fields
        """

        fluxes_h = F.softplus(self.h_flux_conv(features))
        fluxes_mx = F.softplus(self.mx_flux_conv(features))
        fluxes_my = F.softplus(self.my_flux_conv(features))

        # Initialize next states
        next_h = h - fluxes_h.sum(dim=1, keepdim=True)
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        # Single merged loop
        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(fluxes_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))

            next_h = next_h + shifted_h
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my


    def _compute_all_LPP(self, h, mx, my, features):
        """
        LPP mode: Lower bound on h, positive fluxes for mx, my
        """
        # h fluxes (lower-bounded)
        raw_h = self.h_flux_conv(features)
        outflow_percentage = torch.sigmoid(raw_h[:, 0:1])
        distribution_ratios = F.softmax(raw_h[:, 1:], dim=1)

        available = h - self.lower_bound
        outflow_amount = available * outflow_percentage
        neighbor_flows_h = outflow_amount * distribution_ratios

        # mx, my fluxes with softplus (positive only)
        fluxes_mx = F.softplus(self.mx_flux_conv(features))
        fluxes_my = F.softplus(self.my_flux_conv(features))

        # Initialize next states
        next_h = h - outflow_amount
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        # Single merged loop
        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))

            next_h = next_h + shifted_h
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my


if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    batch_size = 4
    H, W = 64, 64

    print(f"\nTest config: batch={batch_size}, H={H}, W={W}")

    for head_config in ['NNN', 'LNN', 'LPP', 'LPP_h']:
        print(f"\n{'=' * 50}")
        print(f"Testing {head_config} mode")
        print('=' * 50)

        model = FluxNet_SW_2D(
            base_channels=32,
            num_blocks=4,
            neighborhood_size=15,
            head_config=head_config
        ).to(device)
        model.eval()

        h = torch.rand(batch_size, 1, H, W).to(device) + 0.5
        mx = torch.randn(batch_size, 1, H, W).to(device) * 0.1
        my = torch.randn(batch_size, 1, H, W).to(device) * 0.1
        input_state = torch.cat([h, mx, my], dim=1)

        with torch.no_grad():
            next_state, h_delta, mx_delta, my_delta = model(input_state)

        # Conservation check
        for i, name in enumerate(['h', 'mx', 'my']):
            initial_mass = input_state[:, i:i + 1].sum()
            final_mass = next_state[:, i:i + 1].sum()
            drift = (final_mass - initial_mass).abs() / (initial_mass.abs() + 1e-8)
            print(f"{name}: initial={initial_mass.item():.6f}, final={final_mass.item():.6f}, drift={drift.item():.2e}")

        # Bound check for h
        h_min = next_state[:, 0].min()
        print(f"h minimum: {h_min.item():.6f} (should be >= 0)")
