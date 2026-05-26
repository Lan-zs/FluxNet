"""
FluxNet-SW 2D: Shallow Water Equations Model with Conservation Constraints

Head configurations:
- PPP: P-head for all three fields (conservative, no bounds)
- LPP: L-head for h (lower-bounded), P-head for mx, my
- LAP: L-head for h + Advection-Pressure decomposition for momentum
      - Momentum advective: carried by water (Δm_adv = Δh * (m/h))
      - Momentum pressure: P-head with h^2 gate (Δm_prs via softplus * h^2)
- PAP: Like LAP but with P-head for h (no lower bound guarantee)
- LAP_no_gate: Like LAP but without h^2 pressure gate

Note: NNN (no constraint) and LNN configurations are kept for backward compatibility
but should be avoided as they don't provide conservation guarantees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad(nn.Module):
    """2D circular padding for periodic boundary conditions"""
    def __init__(self, padding):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')


class DoubleConv(nn.Module):
    """Double convolution block with circular padding"""
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
    Shallow Water Equations Multi-Head Model with Conservation Constraints

    This is the main model for shallow water equations, supporting multiple
    head configurations for different constraint combinations.

    Recommended configuration: LAP (our proposed method)
    - h: L-head (lower-bounded, conservative)
    - mx, my: Advection + Pressure decomposition

    Args:
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolution kernel size
        act_fn: Activation function
        norm_2d: Normalization layer
        neighborhood_size: Size of flux neighborhood (must be odd)
        lower_bound: Lower bound for h field (default 0.0)
        head_config: Head configuration string
            - 'LAP': L-head for h + Advection-Pressure decomposition (RECOMMENDED)
            - 'PAP': P-head for h + Advection-Pressure decomposition (ablation)
            - 'LAP_no_gate': LAP without h^2 pressure gate (ablation)
            - 'PPP': P-head for all fields (ablation)
            - 'LPP': L-head for h, P-head for mx/my (ablation)
            - 'NNN': No constraint (deprecated, for backward compatibility)
    """

    def __init__(self,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 neighborhood_size=15,
                 lower_bound=0.0,
                 head_config='LAP'):
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

        # Head-specific flux convolutions based on configuration
        if head_config in ['LAP', 'LAP_no_gate', 'LPP', 'LNN']:
            # L-head for h: outflow percentage (1) + distribution ratios (num_neighbors)
            self.h_flux_conv = nn.Conv2d(base_channels, 1 + self.num_neighbors, kernel_size=1)
        else:
            # P-head or N-head for h: just neighbor fluxes
            self.h_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)

        # mx, my flux heads
        self.mx_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)
        self.my_flux_conv = nn.Conv2d(base_channels, self.num_neighbors, kernel_size=1)

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
        Forward pass through the model.

        Args:
            x: Input tensor [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: [batch, 3, H, W] - predicted next state
            h_delta: Change in h field
            mx_delta: Change in mx field
            my_delta: Change in my field
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
        if self.head_config == 'LAP':
            next_h, next_mx, next_my = self._compute_LAP(h, mx, my, features)
        elif self.head_config == 'PAP':
            next_h, next_mx, next_my = self._compute_PAP(h, mx, my, features)
        elif self.head_config == 'LAP_no_gate':
            next_h, next_mx, next_my = self._compute_LAP_no_gate(h, mx, my, features)
        elif self.head_config == 'PPP':
            next_h, next_mx, next_my = self._compute_PPP(h, mx, my, features)
        elif self.head_config == 'LPP':
            next_h, next_mx, next_my = self._compute_LPP(h, mx, my, features)
        elif self.head_config == 'NNN':
            next_h, next_mx, next_my = self._compute_NNN(h, mx, my, features)
        else:
            raise ValueError(f"Unknown head_config: {self.head_config}")

        next_state = torch.cat([next_h, next_mx, next_my], dim=1)

        return (next_state,
                next_h - h,
                next_mx - mx,
                next_my - my)

    def _compute_LAP(self, h, mx, my, features):
        """
        LAP mode (RECOMMENDED):
        - h: L-head (lower-bounded, conservative)
        - momentum: Advection + Pressure decomposition
          - Advective: Δm_adv = Δh * (m/h) (momentum carried by water)
          - Pressure: P-head with h^2 gate (pressure flux scaled by water depth squared)
        """
        # ---- 1) h fluxes (L-head: lower-bounded) ----
        raw_h = self.h_flux_conv(features)                       # [B, 1+N, H, W]
        outflow_percentage = torch.sigmoid(raw_h[:, 0:1])        # alpha in [0,1]
        distribution_ratios = F.softmax(raw_h[:, 1:], dim=1)     # pi over neighbors

        available = h - self.lower_bound                         # [B,1,H,W]
        outflow_amount = available * outflow_percentage          # total outflow mass
        neighbor_flows_h = outflow_amount * distribution_ratios  # Δh_{i->j}, [B,N,H,W]

        # h update: subtract outflow first
        next_h = h - outflow_amount

        # ---- 2) Momentum advective part (carried by water) ----
        # u = mx / (h + 1e-6)   # velocity in x [B,1,H,W]
        # v = my / (h + 1e-6)   # velocity in y [B,1,H,W]
        u = mx / (h + 1e-1)   # velocity in x [B,1,H,W]
        v = my / (h + 1e-1)   # velocity in y [B,1,H,W]

        # Advective momentum carried with water transport
        flux_adv_mx = neighbor_flows_h * u   # [B,N,H,W], can be +/- (velocity can be +/-)
        flux_adv_my = neighbor_flows_h * v

        # ---- 3) Pressure/residual part (P-head with h^2 gate) ----
        # Non-negative fluxes via softplus
        flux_prs_mx = F.softplus(self.mx_flux_conv(features))   # [B,N,H,W] >= 0
        flux_prs_my = F.softplus(self.my_flux_conv(features))

        # Gate by h^2: pressure flux scales with water depth squared
        # This ensures dry cells (h~0) don't emit pressure flux
        press_scale = torch.clamp(h, min=0.0) ** 2

        flux_prs_mx = flux_prs_mx * press_scale
        flux_prs_my = flux_prs_my * press_scale

        # ---- 4) Total momentum flux = advective + pressure ----
        fluxes_mx = flux_adv_mx + flux_prs_mx
        fluxes_my = flux_adv_my + flux_prs_my

        # ---- 5) Conservative update ----
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        # Add incoming fluxes from neighbors (single merged loop for efficiency)
        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            # Incoming h from neighbor
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_h = next_h + shifted_h

            # Incoming momentum from neighbors
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my

    def _compute_PAP(self, h, mx, my, features):
        """
        PAP mode (ablation without L-head for h):
        - h: P-head (soft positive fluxes, no lower bound guarantee)
        - momentum: Same Advection + Pressure decomposition as LAP

        Note: Unlike LAP, PAP can produce negative h values because there's no
        bound on the outflow. To prevent numerical instability:
        - Clamp h when computing velocities
        - Limit the total outflow to available h (soft constraint via sigmoid scaling)
        """
        # ---- 1) h fluxes (P-head with scaled output to prevent excessive outflow) ----
        raw_h_fluxes = self.h_flux_conv(features)  # [B,N,H,W]

        # Use sigmoid to get distribution ratios, then scale by available h
        # This provides a soft constraint on outflow while not strictly enforcing lower bound
        h_safe = torch.clamp(h, min=1e-6)  # Ensure h is positive for computation
        flux_ratios = torch.sigmoid(raw_h_fluxes)  # Each neighbor gets a fraction in [0,1]

        # Scale by available h per neighbor (total outflow can exceed h, but is regulated)
        scale_factor = h_safe / (self.num_neighbors + 1)  # Average budget per neighbor
        neighbor_flows_h = flux_ratios * scale_factor

        # h update: subtract outflow, add inflow
        next_h = h - neighbor_flows_h.sum(dim=1, keepdim=True)

        # ---- 2) Momentum advective part ----
        # Use clamped h to avoid division by very small/negative numbers
        h_for_velocity = torch.clamp(h, min=1e-4)
        u = mx / h_for_velocity
        v = my / h_for_velocity

        # Clamp velocity to prevent extreme values
        u = torch.clamp(u, min=-10.0, max=10.0)
        v = torch.clamp(v, min=-10.0, max=10.0)

        flux_adv_mx = neighbor_flows_h * u
        flux_adv_my = neighbor_flows_h * v

        # ---- 3) Pressure part with h^2 gate ----
        flux_prs_mx = F.softplus(self.mx_flux_conv(features))
        flux_prs_my = F.softplus(self.my_flux_conv(features))
        press_scale = torch.clamp(h, min=0.0) ** 2
        flux_prs_mx = flux_prs_mx * press_scale
        flux_prs_my = flux_prs_my * press_scale

        # ---- 4) Total momentum flux ----
        fluxes_mx = flux_adv_mx + flux_prs_mx
        fluxes_my = flux_adv_my + flux_prs_my

        # ---- 5) Conservative update ----
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_h = next_h + shifted_h
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my

    def _compute_LAP_no_gate(self, h, mx, my, features):
        """
        LAP_no_gate mode (ablation without h^2 pressure gate):
        - h: L-head (lower-bounded)
        - momentum: Advection + Pressure WITHOUT h^2 gate
        """
        # ---- 1) h fluxes (L-head) ----
        raw_h = self.h_flux_conv(features)
        outflow_percentage = torch.sigmoid(raw_h[:, 0:1])
        distribution_ratios = F.softmax(raw_h[:, 1:], dim=1)
        available = h - self.lower_bound
        outflow_amount = available * outflow_percentage
        neighbor_flows_h = outflow_amount * distribution_ratios

        next_h = h - outflow_amount

        # ---- 2) Momentum advective part ----
        u = mx / (h + 1e-6)
        v = my / (h + 1e-6)
        flux_adv_mx = neighbor_flows_h * u
        flux_adv_my = neighbor_flows_h * v

        # ---- 3) Pressure part WITHOUT h^2 gate ----
        flux_prs_mx = F.softplus(self.mx_flux_conv(features))
        flux_prs_my = F.softplus(self.my_flux_conv(features))
        # No press_scale multiplication here!

        # ---- 4) Total momentum flux ----
        fluxes_mx = flux_adv_mx + flux_prs_mx
        fluxes_my = flux_adv_my + flux_prs_my

        # ---- 5) Conservative update ----
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_h = next_h + shifted_h
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my

    def _compute_PPP(self, h, mx, my, features):
        """
        PPP mode (ablation): All three fields use P-head
        - Conservative but no bound guarantees
        """
        fluxes_h = F.softplus(self.h_flux_conv(features))
        fluxes_mx = F.softplus(self.mx_flux_conv(features))
        fluxes_my = F.softplus(self.my_flux_conv(features))

        # Initialize next states (subtract outflow)
        next_h = h - fluxes_h.sum(dim=1, keepdim=True)
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        # Add incoming fluxes (single merged loop)
        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(fluxes_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))

            next_h = next_h + shifted_h
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my

    def _compute_LPP(self, h, mx, my, features):
        """
        LPP mode (ablation): L-head for h, P-head for mx/my
        - h is lower-bounded and conservative
        - mx, my are conservative but can become very negative
        """
        # h fluxes (L-head: lower-bounded)
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

        # Add incoming fluxes (single merged loop)
        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(neighbor_flows_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_mx = torch.roll(fluxes_mx[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            shifted_my = torch.roll(fluxes_my[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))

            next_h = next_h + shifted_h
            next_mx = next_mx + shifted_mx
            next_my = next_my + shifted_my

        return next_h, next_mx, next_my

    def _compute_NNN(self, h, mx, my, features):
        """
        NNN mode (deprecated): No constraints, fluxes can be positive or negative
        - NOT recommended: can cause unbounded growth
        """
        fluxes_h = self.h_flux_conv(features)  # No activation
        fluxes_mx = self.mx_flux_conv(features)
        fluxes_my = self.my_flux_conv(features)

        next_h = h - fluxes_h.sum(dim=1, keepdim=True)
        next_mx = mx - fluxes_mx.sum(dim=1, keepdim=True)
        next_my = my - fluxes_my.sum(dim=1, keepdim=True)

        for n, (dh, dw) in enumerate(self.neighbor_offsets_list):
            shifted_h = torch.roll(fluxes_h[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
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

    # Test all head configurations
    for head_config in ['LAP', 'PAP', 'LAP_no_gate', 'PPP', 'LPP']:
        print(f"\n{'=' * 60}")
        print(f"Testing {head_config} mode")
        print('=' * 60)

        model = FluxNet_SW_2D(
            base_channels=32,
            num_blocks=4,
            neighborhood_size=7,
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
        print("Conservation check:")
        for i, name in enumerate(['h', 'mx', 'my']):
            initial_mass = input_state[:, i:i + 1].sum()
            final_mass = next_state[:, i:i + 1].sum()
            drift = (final_mass - initial_mass).abs() / (initial_mass.abs() + 1e-8)
            status = "OK" if drift.item() < 1e-5 else "DRIFT"
            print(f"  {name}: initial={initial_mass.item():.4f}, final={final_mass.item():.4f}, "
                  f"drift={drift.item():.2e} [{status}]")

        # Bound check for h
        h_min = next_state[:, 0].min()
        h_bound_ok = h_min.item() >= -1e-6
        print(f"  h minimum: {h_min.item():.6f} (bound satisfied: {h_bound_ok})")

    print("\n" + "=" * 60)
    print("All tests completed!")
