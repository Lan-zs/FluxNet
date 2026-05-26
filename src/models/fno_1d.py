"""
FNO 1D: Fourier Neural Operator for 1D problems

This is a baseline model without structural conservation guarantees.
Supports:
- Direct or residual prediction mode
- Optional bound constraints via sigmoid/softplus post-processing
- Soft conservation loss during training (external)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer (Fourier Layer)"""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        length = x.shape[-1]

        # Compute 1D FFT
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, length // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Only multiply the first 'modes' modes
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Compute inverse 1D FFT
        x = torch.fft.irfft(out_ft, n=length)

        return x


class FNOBlock1d(nn.Module):
    """Single FNO Block: Spectral Conv + Local Conv + Activation"""

    def __init__(self, channels, modes, act_fn=nn.GELU):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = act_fn()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        return self.act(x1 + x2)


class FNO_1D(nn.Module):
    """
    1D Fourier Neural Operator

    No structural conservation guarantees - serves as a baseline.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep
        width: Width of hidden channels
        num_layers: Number of FNO blocks
        prediction_mode: 'direct' or 'residual'
        bound_mode: 'none', 'lower', 'double'
            - 'none': No bound constraint
            - 'lower': Output >= lower_bound via softplus
            - 'double': lower_bound <= output <= upper_bound via sigmoid
        lower_bound: Lower bound value
        upper_bound: Upper bound value
    """

    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 base_channels=64,
                 num_blocks=1,
                 kernel_size=3,
                 modes=16,
                 width=64,
                 num_layers=4,
                 prediction_mode='residual',
                 bound_mode='none',
                 lower_bound=0.0,
                 upper_bound=1.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.prediction_mode = prediction_mode
        self.bound_mode = bound_mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Lift to higher dimension
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes)
            for _ in range(num_layers)
        ])

        # Project back to output dimension
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, length]

        Returns:
            next_field: tuple of ([batch, out_channels, length],)
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
            next_field = x[:, 0:self.out_channels] + output
        else:
            next_field = output

        # Apply bound constraints
        if self.bound_mode == 'lower':
            next_field = F.softplus(next_field - self.lower_bound) + self.lower_bound
        elif self.bound_mode == 'double':
            range_size = self.upper_bound - self.lower_bound
            next_field = torch.sigmoid(next_field) * range_size + self.lower_bound

        return (next_field,)


class FNO_FluxD_1D(nn.Module):
    """
    FNO with FluxNet-D head for 1D problems

    Uses FNO as feature extractor, then applies FluxNet-D conservation head.
    This ensures exact conservation while leveraging FNO's spectral capabilities.

    Note: This limits FNO's cross-resolution generalization ability.

    Args:
        in_channels: Number of input channels
        modes: Number of Fourier modes
        width: Width of hidden channels
        num_layers: Number of FNO blocks
        neighborhood_size: Size of flux neighborhood (must be odd)
        lower_bound: Lower bound for the conserved field
        upper_bound: Upper bound for the conserved field
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=1,
                 kernel_size=3,
                 modes=16,
                 width=64,
                 num_layers=4,
                 neighborhood_size=15,
                 lower_bound=0.0,
                 upper_bound=1.0):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"

        self.modes = modes
        self.width = width
        self.neighborhood_size = neighborhood_size
        self.num_neighbors = neighborhood_size - 1

        # Bounds
        self.register_buffer('lower_bound_value', torch.tensor(lower_bound))
        self.register_buffer('upper_bound_value', torch.tensor(upper_bound))

        # Total channels: 2 sets of (1 percentage + num_neighbors distribution)
        self.total_channels = 2 * (1 + self.num_neighbors)

        # Lift to higher dimension
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes)
            for _ in range(num_layers)
        ])

        # Intermediate projection to adapt FNO features to flux prediction
        self.flux_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
        )

        # FluxNet-D head: predict flux parameters for both approaches
        self.flux_conv = nn.Conv1d(width // 2, self.total_channels, kernel_size=1)

        # Initialize flux_conv weights to small values for stable start
        nn.init.normal_(self.flux_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.flux_conv.bias)

        # Generate neighbor offsets
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            if i != 0:
                neighbor_offsets.append(i)
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    @property
    def lower_bound(self):
        return self.lower_bound_value

    @property
    def upper_bound(self):
        return self.upper_bound_value

    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, length]

        Returns:
            next_field: [batch, 1, length]
            outflow_change: [batch, 1, length]
            inflow_change: [batch, 1, length]
        """
        # FNO feature extraction
        features = self.lift(x)
        for block in self.fno_blocks:
            features = block(features)

        # Project features for flux prediction
        flux_features = self.flux_proj(features)

        # Predict flux parameters for both approaches
        raw_fluxes = self.flux_conv(flux_features)

        # Split for outflow approach (lower bound)
        outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])
        outflow_distribution = F.softmax(raw_fluxes[:, 1:self.num_neighbors+1], dim=1)

        # Split for inflow approach (upper bound)
        inflow_percentage = torch.sigmoid(raw_fluxes[:, self.num_neighbors+1:self.num_neighbors+2])
        inflow_distribution = F.softmax(raw_fluxes[:, self.num_neighbors+2:], dim=1)

        # Get conserved field
        conserved_field = x[:, 0:1]

        # Compute changes from both approaches
        outflow_change, inflow_change = self._compute_transport(
            conserved_field,
            outflow_percentage,
            outflow_distribution,
            inflow_percentage,
            inflow_distribution
        )

        # Average the two approaches
        combined_change = (outflow_change + inflow_change) / 2
        next_field = conserved_field + combined_change

        return next_field, outflow_change, inflow_change

    def _compute_transport(self, current_field, outflow_pct, outflow_dist, inflow_pct, inflow_dist):
        """Compute conservative transport using dual approach"""
        # Outflow approach: based on available amount above lower bound
        available_outflow = current_field - self.lower_bound
        outflow_amount = available_outflow * outflow_pct
        outflow_change = -outflow_amount

        # Distribute outflow
        outflow_to_neighbors = outflow_amount * outflow_dist

        # Inflow approach: based on available capacity below upper bound
        available_inflow = self.upper_bound - current_field
        inflow_amount = available_inflow * inflow_pct
        inflow_change = inflow_amount

        # Distribute inflow
        inflow_from_neighbors = inflow_amount * inflow_dist

        # Add contributions from shifted neighbors
        for n, offset in enumerate(self.neighbor_offsets):
            # Outflow: shift in opposite direction
            shifted_out = torch.roll(outflow_to_neighbors[:, n:n+1], shifts=-int(offset), dims=2)
            outflow_change = outflow_change + shifted_out

            # Inflow: shift in same direction (pulling from neighbors)
            shifted_in = torch.roll(inflow_from_neighbors[:, n:n+1], shifts=int(offset), dims=2)
            inflow_change = inflow_change - shifted_in

        return outflow_change, inflow_change


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    batch_size = 4
    length = 64

    # Test FNO_1D
    print("\n=== Testing FNO_1D (no bound) ===")
    model = FNO_1D(in_channels=2, out_channels=1, modes=16, width=32).to(device)
    x = torch.randn(batch_size, 2, length).to(device)
    out, = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test FNO_1D with double bound
    print("\n=== Testing FNO_1D (double bound) ===")
    model_bound = FNO_1D(in_channels=2, bound_mode='double').to(device)
    out_bound, = model_bound(x)
    print(f"Output range: [{out_bound.min().item():.4f}, {out_bound.max().item():.4f}]")

    # Test FNO_FluxD_1D
    print("\n=== Testing FNO_FluxD_1D ===")
    model_flux = FNO_FluxD_1D(in_channels=2, modes=16, width=32, neighborhood_size=15).to(device)
    model_flux.eval()

    conserved = torch.rand(batch_size, 1, length).to(device) * 0.6 + 0.2  # [0.2, 0.8]
    external = torch.randn(batch_size, 1, length).to(device)
    x_flux = torch.cat([conserved, external], dim=1)

    with torch.no_grad():
        next_field, outflow, inflow = model_flux(x_flux)

    # Conservation check
    initial_mass = conserved.sum()
    final_mass = next_field.sum()
    drift = (final_mass - initial_mass).abs() / (initial_mass.abs() + 1e-8)
    print(f"Conservation drift: {drift.item():.2e}")
    print(f"Output range: [{next_field.min().item():.4f}, {next_field.max().item():.4f}]")
    print(f"Parameters: {sum(p.numel() for p in model_flux.parameters()):,}")

    print("\nAll tests completed!")
