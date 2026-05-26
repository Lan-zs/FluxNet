
"""
FNO-FluxLAP for Shallow Water Equations

FNO as backbone (feature extraction) + LAP conservation head.

LAP Head Configuration:
- L-head for h: conservation + lower bound (h >= lower_bound)
- Advection-Pressure decomposition for momentum:
  - Advective: Δm_adv = Δh * (m/h)  (carried by water)
  - Pressure: P-head with h^2 gating

Note: Using FNO as backbone limits cross-resolution generalization.
This is for ablation study purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer (Fourier Layer)

    Note: FFT inherently assumes periodic boundary conditions,
    so this layer naturally preserves periodicity.
    """

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
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute 2D FFT (periodic by nature)
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


class FNO_FluxLAP(nn.Module):
    """
    FNO backbone + LAP Conservation Head for Shallow Water Equations

    Uses FNO for feature extraction, then applies LAP (L-head + Advection-Pressure)
    conservation head to ensure exact conservation and bounded predictions.

    LAP head mechanism:
    - L-head for h: lower bound constraint (h >= lower_bound) + exact conservation
    - Advection-Pressure decomposition for momentum:
        - Advective: Δm_adv = Δh * (m/h) (momentum carried with water)
        - Pressure: P-head (softplus) * h^2 gating

    Note: FNO backbone limits cross-resolution generalization ability.
    This model is for ablation study to show that the LAP head design
    can work with different feature extractors.

    Args:
        modes1: Number of Fourier modes in first spatial dimension
        modes2: Number of Fourier modes in second spatial dimension
        width: Width of hidden channels
        num_layers: Number of FNO blocks
        in_channels: Number of input channels (default 3: h, mx, my)
        neighborhood_size: Size of flux neighborhood (must be odd)
        lower_bound: Lower bound for h field (default 0.0)
    """

    def __init__(self,
                 modes1=16,
                 modes2=16,
                 width=64,
                 num_layers=4,
                 in_channels=3,
                 neighborhood_size=7,
                 lower_bound=0.0,
                 prediction_mode='residual'):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.neighborhood_size = neighborhood_size
        self.num_neighbors = neighborhood_size * neighborhood_size - 1

        self.register_buffer('lower_bound', torch.tensor(lower_bound))

        # Lift to higher dimension (1x1 conv, no boundary issues)
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # FNO blocks (periodic by FFT nature)
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2)
            for _ in range(num_layers)
        ])

        # ===== Flux Heads =====
        # L-head for h: outflow percentage (1) + distribution ratios (num_neighbors)
        self.h_flux_conv = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, 1 + self.num_neighbors, kernel_size=1)
        )

        # P-head for mx: positive fluxes to neighbors
        self.mx_flux_conv = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, self.num_neighbors, kernel_size=1)
        )

        # P-head for my: positive fluxes to neighbors
        self.my_flux_conv = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, self.num_neighbors, kernel_size=1)
        )

        # Pre-compute neighbor offsets for flux exchange
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
            x: [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: tuple of ([batch, 3, H, W],)
        """
        # Lift to higher dimension
        x_lifted = self.lift(x)

        # FNO blocks for feature extraction
        for block in self.fno_blocks:
            x_lifted = block(x_lifted)

        # Extract current state
        h = x[:, 0:1]
        mx = x[:, 1:2]
        my = x[:, 2:3]

        # Compute conservative update via flux mechanism (LAP head)
        next_h, next_mx, next_my = self._compute_all_LAP(h, mx, my, x_lifted)

        next_state = torch.cat([next_h, next_mx, next_my], dim=1)

        return (next_state,)

    def _compute_all_LAP(self, h, mx, my, features):
        """
        LAP_advP mode:
        - h: L-head (lower-bounded, conservative) -> produces neighbor_flows_h = Δh_{i->j}
        - momentum advective (carried by water): Δm_adv_{i->j} = Δh_{i->j} * (m/h)
        - momentum pressure/residual: P-head (softplus) -> Δm_prs_{i->j} >= 0
        - total momentum flux: Δm = Δm_adv + Δm_prs
        """
        # ---- 1) h fluxes (lower-bounded) ----
        raw_h = self.h_flux_conv(features)                       # [B, 1+N, H, W]
        outflow_percentage = torch.sigmoid(raw_h[:, 0:1])        # alpha in [0,1]
        distribution_ratios = F.softmax(raw_h[:, 1:], dim=1)     # pi over neighbors

        available = h - self.lower_bound                         # [B,1,H,W]
        outflow_amount = available * outflow_percentage          # total outflow mass from each cell
        neighbor_flows_h = outflow_amount * distribution_ratios  # Δh_{i->j}, [B,N,H,W]

        # h update: subtract outflow first
        next_h = h - outflow_amount

        # ---- 2) momentum advective part (follows water transport plan) ----
        u = mx / (h + 1e-6)   # [B,1,H,W]
        v = my / (h + 1e-6)   # [B,1,H,W]

        # advective momentum carried with water:
        flux_adv_mx = neighbor_flows_h * u   # [B,N,H,W], can be +/- because u can be +/-
        flux_adv_my = neighbor_flows_h * v

        # ---- 3) pressure/residual part learned by P-head ----
        # nonnegative "outgoing magnitudes" per neighbor direction
        flux_prs_mx = F.softplus(self.mx_flux_conv(features))   # [B,N,H,W] >= 0
        flux_prs_my = F.softplus(self.my_flux_conv(features))

        # IMPORTANT (recommended):
        # Gate/scale pressure flux by a physically meaningful scale so dry cells do not emit pressure flux.
        press_scale = torch.clamp(h, min=0.0) ** 2

        flux_prs_mx = flux_prs_mx * press_scale
        flux_prs_my = flux_prs_my * press_scale

        # ---- 4) total momentum flux ----
        fluxes_mx = flux_adv_mx + flux_prs_mx
        fluxes_my = flux_adv_my + flux_prs_my

        # ---- 5) conservative update for mx,my (same as your pattern) ----
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            # incoming h
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_h = next_h + shifted_h

            # incoming mx,my (inflow from neighbors)
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing FNO_FluxLAP on: {device}")
    print("=" * 60)

    batch_size = 4
    H, W = 64, 64

    model = FNO_FluxLAP(
        modes1=16,
        modes2=16,
        width=32,
        num_layers=4,
        neighborhood_size=7,
        lower_bound=0.0
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Neighborhood size: {model.neighborhood_size}x{model.neighborhood_size}")
    print(f"Number of neighbors: {model.num_neighbors}")
    print("=" * 60)

    # Create input
    h = torch.rand(batch_size, 1, H, W).to(device) + 0.5  # h > 0
    mx = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    my = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    input_state = torch.cat([h, mx, my], dim=1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output, = model(input_state)

    print(f"Input shape: {input_state.shape}")
    print(f"Output shape: {output.shape}")
    print("=" * 60)

    # Conservation check (should be perfectly conserved)
    print("Conservation Check (should be ~0 for LAP):")
    for i, name in enumerate(['h', 'mx', 'my']):
        initial_sum = input_state[:, i].sum()
        final_sum = output[:, i].sum()
        absolute_drift = (final_sum - initial_sum).abs()
        relative_drift = absolute_drift / (initial_sum.abs() + 1e-8)
        print(f"  {name}: absolute drift = {absolute_drift.item():.2e}, "
              f"relative drift = {relative_drift.item():.2e}")

    print("=" * 60)

    # Lower bound check for h
    h_min = output[:, 0].min().item()
    print(f"Lower Bound Check:")
    print(f"  h minimum value: {h_min:.6f}")
    print(f"  Lower bound: {model.lower_bound.item():.6f}")
    print(f"  Constraint satisfied: {h_min >= model.lower_bound.item()}")

    print("=" * 60)

    # Multi-step test
    print("Multi-step Conservation Test (10 steps):")
    state = input_state.clone()
    with torch.no_grad():
        for step in range(10):
            state, = model(state)

    for i, name in enumerate(['h', 'mx', 'my']):
        initial_sum = input_state[:, i].sum()
        final_sum = state[:, i].sum()
        relative_drift = (final_sum - initial_sum).abs() / (initial_sum.abs() + 1e-8)
        print(f"  {name} drift after 10 steps: {relative_drift.item():.2e}")

    print("=" * 60)
    print("Test completed successfully!")
