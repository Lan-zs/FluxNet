"""
FluxNet-D 1D: Dual-Bounded Conservative Flux Network for 1D problems

This model:
- Guarantees mass conservation through flux-based updates
- Enforces both lower and upper bound constraints (e.g., 0 <= rho <= 1)
- Suitable for doubly-bounded conserved quantities (e.g., traffic density, phase field)
- Uses dual approach: outflow from lower bound + inflow to upper bound
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad1D(nn.Module):
    """1D circular padding for periodic boundary conditions"""
    def __init__(self, padding):
        super(CircularPad1D, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding), mode='circular')


class DoubleConv1D(nn.Module):
    """Double convolution block with circular padding"""
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, norm_1d=nn.BatchNorm1d):
        super(DoubleConv1D, self).__init__()
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


class FluxNet_D_1D(nn.Module):
    """
    1D Dual-Bounded Flux Network

    Uses two approaches simultaneously:
    - Outflow approach: based on (current - lower_bound)
    - Inflow approach: based on (upper_bound - current)
    Final prediction is the average of both approaches

    Args:
        in_channels: Number of input channels (conserved field + external fields)
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for convolutions
        act_fn: Activation function
        norm_1d: Normalization layer
        neighborhood_size: Size of the neighborhood stencil (must be odd)
        lower_bound: Lower bound for the conserved field
        upper_bound: Upper bound for the conserved field
        learnable_lower_bound: Whether lower bound is learnable
        learnable_upper_bound: Whether upper bound is learnable
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_1d=nn.BatchNorm1d,
                 neighborhood_size=15,
                 lower_bound=0.0,
                 upper_bound=1.0,
                 learnable_lower_bound=False,
                 learnable_upper_bound=False):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"

        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.learnable_lower_bound = learnable_lower_bound
        self.learnable_upper_bound = learnable_upper_bound

        # Setup bounds (learnable or fixed)
        if learnable_lower_bound:
            logit_lower = self._inverse_sigmoid(lower_bound)
            self.lower_bound_logit = nn.Parameter(logit_lower.data)
        else:
            self.register_buffer('lower_bound_value', torch.tensor(lower_bound).detach())

        if learnable_upper_bound:
            logit_upper = self._inverse_sigmoid(upper_bound)
            self.upper_bound_logit = nn.Parameter(logit_upper.data)
        else:
            self.register_buffer('upper_bound_value', torch.tensor(upper_bound).detach())

        # Number of neighbors
        self.num_neighbors = neighborhood_size - 1

        # Total channels: 2 sets of (1 percentage + num_neighbors distribution)
        self.total_channels = 2 * (1 + self.num_neighbors)

        # First convolution layer
        self.first_conv = nn.Sequential(
            CircularPad1D(kernel_size // 2),
            nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=0),
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

        # Flux prediction layer for both approaches
        self.flux_conv = nn.Conv1d(base_channels, self.total_channels, kernel_size=1)

        # Generate neighbor offsets
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            if i != 0:
                neighbor_offsets.append(i)
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-7):
        """Inverse sigmoid for parameter initialization"""
        x = torch.clamp(torch.tensor(x), eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def lower_bound(self):
        """Get current lower bound"""
        if self.learnable_lower_bound:
            return torch.sigmoid(self.lower_bound_logit)
        else:
            return self.lower_bound_value

    @property
    def upper_bound(self):
        """Get current upper bound"""
        if self.learnable_upper_bound:
            return torch.sigmoid(self.upper_bound_logit)
        else:
            return self.upper_bound_value

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, in_channels, length]

        Returns:
            next_field: [batch, 1, length]
            outflow_change: [batch, 1, length]
            inflow_change: [batch, 1, length]
        """
        # Extract features
        features = self.first_conv(x)

        # Process through residual blocks
        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # Predict flux parameters for both approaches
        raw_fluxes = self.flux_conv(features)

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
        """
        Compute conservative transport using dual approach

        Args:
            current_field: [batch, 1, length]
            outflow_pct, outflow_dist: Outflow parameters
            inflow_pct, inflow_dist: Inflow parameters

        Returns:
            outflow_change: [batch, 1, length]
            inflow_change: [batch, 1, length]
        """
        # Outflow approach: based on available amount above lower bound
        available_outflow = current_field - self.lower_bound
        outflow_amount = available_outflow * outflow_pct
        outflow_change = -outflow_amount

        # Distribute outflow
        outflow_to_neighbors = outflow_amount * outflow_dist  # [batch, num_neighbors, length]

        # Inflow approach: based on available capacity below upper bound
        available_inflow = self.upper_bound - current_field
        inflow_amount = available_inflow * inflow_pct
        inflow_change = inflow_amount

        # Distribute inflow
        inflow_from_neighbors = inflow_amount * inflow_dist  # [batch, num_neighbors, length]

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
    """Test FluxNet_D_1D for conservation and dual bounds"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create model
    lower_bound = 0.0
    upper_bound = 1.0
    model = FluxNet_D_1D(
        in_channels=2,
        base_channels=32,
        num_blocks=4,
        kernel_size=3,
        neighborhood_size=15,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    ).to(device)

    model.eval()

    print("\n=== Testing Conservation and Dual Bounds ===")
    batch_size = 4
    length = 64

    # Create random input within bounds
    conserved_field = torch.rand(batch_size, 1, length).to(device) * 0.6 + 0.2  # [0.2, 0.8]
    external_field = torch.randn(batch_size, 1, length).to(device)
    input_tensor = torch.cat([conserved_field, external_field], dim=1)

    # Forward pass
    with torch.no_grad():
        next_field, outflow_change, inflow_change = model(input_tensor)

    # Check conservation
    initial_mass = conserved_field.sum()
    final_mass = next_field.sum()
    mass_diff = (final_mass - initial_mass).abs()
    relative_error = mass_diff / (initial_mass + 1e-8)

    print(f"Initial mass: {initial_mass.item():.6f}")
    print(f"Final mass: {final_mass.item():.6f}")
    print(f"Mass difference: {mass_diff.item():.10f}")
    print(f"Relative error: {relative_error.item():.10f}")

    # Check bounds
    min_val = next_field.min().item()
    max_val = next_field.max().item()
    print(f"\nBound constraints: [{lower_bound}, {upper_bound}]")
    print(f"Output range: [{min_val:.6f}, {max_val:.6f}]")

    # Check approach consistency
    approach_diff = (outflow_change - inflow_change).abs().mean()
    print(f"\nAverage difference between approaches: {approach_diff.item():.6f}")

    # Validation
    conservation_ok = relative_error < 1e-5
    lower_ok = min_val >= lower_bound - 1e-6
    upper_ok = max_val <= upper_bound + 1e-6

    print(f"\n{'✓' if conservation_ok else '✗'} Conservation test {'PASSED' if conservation_ok else 'FAILED'}")
    print(f"{'✓' if lower_ok else '✗'} Lower bound test {'PASSED' if lower_ok else 'FAILED'}")
    print(f"{'✓' if upper_ok else '✗'} Upper bound test {'PASSED' if upper_ok else 'FAILED'}")

    # Multi-step test
    print("\n=== Multi-Step Test ===")
    current = conserved_field.clone()
    initial_mass_total = current.sum()

    for step in range(10):
        input_tensor = torch.cat([current, external_field], dim=1)
        with torch.no_grad():
            current, _, _ = model(input_tensor)

        step_mass = current.sum()
        step_min = current.min()
        step_max = current.max()
        drift = (step_mass - initial_mass_total).abs() / (initial_mass_total + 1e-8)

        bounds_ok = (step_min >= lower_bound - 1e-6) and (step_max <= upper_bound + 1e-6)
        status = "✓" if bounds_ok else "✗"

        print(f"Step {step+1}: {status} mass={step_mass.item():.6f}, drift={drift.item():.10f}, "
              f"range=[{step_min.item():.6f}, {step_max.item():.6f}]")

    print("\nAll tests completed!")
