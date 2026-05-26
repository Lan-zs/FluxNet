"""
FluxNet-P 2D: Positive-flux Conservative Flux Network for 2D problems

This model:
- Guarantees mass conservation through flux-based updates
- Uses softplus to ensure all fluxes are positive (P = Positive)
- Defines a consistent positive flux direction
- Suitable for problems where directional flow constraints matter
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


class FluxNet_P(nn.Module):
    """
    2D Positive-flux Flux Network

    Args:
        in_channels: Number of input channels
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for convolutions
        act_fn: Activation function
        norm_2d: Normalization layer
        neighborhood_size: Size of the neighborhood stencil (must be odd)
    """

    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 neighborhood_size=15):
        super().__init__()

        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.num_neighbors = neighborhood_size * neighborhood_size - 1

        # First convolution layer
        self.first_conv = nn.Sequential(
            CircularPad(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Flux prediction layer
        self.flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)

        # Generate neighbor offsets
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i != 0 or j != 0:
                    neighbor_offsets.append((i, j))
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, in_channels, H, W]

        Returns:
            next_field: Updated field [batch, 1, H, W]
            delta_field: Change in field [batch, 1, H, W]
        """
        # Extract features
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # Predict fluxes
        raw_fluxes = self.flux_conv(features)

        # Apply softplus to ensure positive fluxes
        fluxes = F.softplus(raw_fluxes)

        # Get conserved field
        conserved_field = x[:, 0:1]

        # Compute transport
        next_field = self._compute_transport(conserved_field, fluxes)

        return next_field, next_field - conserved_field

    def _compute_transport(self, current_field, fluxes):
        """Conservative transport with positive fluxes"""
        next_field = current_field.clone()

        # Subtract total outgoing flux
        total_outgoing = fluxes.sum(dim=1, keepdim=True)
        next_field = next_field - total_outgoing

        # Add incoming flux from neighbors
        for n, (dh, dw) in enumerate(self.neighbor_offsets):
            neighbor_flux = fluxes[:, n:n+1]
            shifted_flux = torch.roll(neighbor_flux, shifts=(-dh, -dw), dims=(2, 3))
            next_field = next_field + shifted_flux

        return next_field


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing FluxNet_P on: {device}")

    model = FluxNet_P(in_channels=1, base_channels=32, num_blocks=4, neighborhood_size=15).to(device)
    model.eval()

    batch_size = 2
    H, W = 48, 48

    input_tensor = torch.rand(batch_size, 1, H, W).to(device)
    initial_mass = input_tensor.sum()

    with torch.no_grad():
        next_field, delta_field = model(input_tensor)

    final_mass = next_field.sum()
    mass_diff = (final_mass - initial_mass).abs()
    relative_error = mass_diff / (initial_mass.abs() + 1e-8)

    print(f"Initial mass: {initial_mass.item():.6f}")
    print(f"Final mass: {final_mass.item():.6f}")
    print(f"Relative error: {relative_error.item():.10f}")

    if relative_error < 1e-5:
        print("Conservation test PASSED")
    else:
        print("Conservation test FAILED")
