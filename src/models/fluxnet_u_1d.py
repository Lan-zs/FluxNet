"""
FluxNet-U 1D: Upper-Bounded Conservative Flux Network for 1D problems

This model:
- Guarantees mass conservation through flux-based updates
- Enforces upper bound constraint (e.g., density <= 1)
- Uses inflow approach: predicts how much each cell can receive from neighbors
- Suitable for upper-bounded conserved quantities

This is related to but distinct from FluxNet-D:
- FluxNet-D uses dual approach (outflow + inflow averaged)
- FluxNet-U uses only inflow approach

For problems with only upper bound (no lower bound), use this model.
For problems with both bounds, use FluxNet-D.
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


class FluxNet_U_1D(nn.Module):
    """
    1D Upper-Bounded Flux Network

    Uses inflow approach: each cell determines how much it receives from neighbors.
    This ensures the cell never exceeds the upper bound.

    Key idea:
    - available_capacity = upper_bound - current_value
    - Each cell can receive up to available_capacity * inflow_percentage
    - The inflow is distributed among neighbors via softmax weights

    Args:
        in_channels: Number of input channels (conserved field + external fields)
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for convolutions
        act_fn: Activation function
        norm_1d: Normalization layer
        neighborhood_size: Size of the neighborhood stencil (must be odd)
        upper_bound: Upper bound for the conserved field
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_1d=nn.BatchNorm1d,
                 neighborhood_size=15,
                 upper_bound=1.0):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"

        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.register_buffer('upper_bound_value', torch.tensor(upper_bound))

        # Number of neighbors (excluding center point)
        self.num_neighbors = neighborhood_size - 1

        # Total channels: 1 for inflow percentage + num_neighbors for distribution
        self.total_channels = 1 + self.num_neighbors

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

        # Flux prediction layer
        self.flux_conv = nn.Conv1d(base_channels, self.total_channels, kernel_size=1)

        # Generate neighbor offsets
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            if i != 0:
                neighbor_offsets.append(i)
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    @property
    def upper_bound(self):
        """Get current upper bound value"""
        return self.upper_bound_value

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, in_channels, length]
               First channel is the conserved field

        Returns:
            next_field: Updated field [batch, 1, length]
            delta_field: Change in field [batch, 1, length]
        """
        # Extract features
        features = self.first_conv(x)

        # Process through residual blocks
        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # Predict flux parameters
        raw_fluxes = self.flux_conv(features)  # [batch, total_channels, length]

        # Split into inflow percentage and distribution ratios
        inflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])  # [batch, 1, length]
        distribution_logits = raw_fluxes[:, 1:]  # [batch, num_neighbors, length]
        distribution_ratios = F.softmax(distribution_logits, dim=1)

        # Get conserved field
        conserved_field = x[:, 0:1]  # [batch, 1, length]

        # Compute transport with upper bound constraint
        next_field = self._compute_transport(
            conserved_field,
            inflow_percentage,
            distribution_ratios
        )

        return next_field, next_field - conserved_field

    def _compute_transport(self, current_field, inflow_percentage, distribution_ratios):
        """
        Compute conservative transport with upper bound constraint using inflow approach

        The key idea: each cell can receive up to (upper_bound - current) amount

        Args:
            current_field: [batch, 1, length]
            inflow_percentage: [batch, 1, length] - fraction of capacity to receive
            distribution_ratios: [batch, num_neighbors, length] - where inflow comes from

        Returns:
            next_field: [batch, 1, length], guaranteed to be <= upper_bound
        """
        # Available capacity for inflow (below upper bound)
        available_capacity = self.upper_bound - current_field

        # Total inflow amount (what this cell will receive)
        inflow_amount = available_capacity * inflow_percentage

        # Start with current field plus inflow
        next_field = current_field + inflow_amount

        # The inflow must come from neighbors (subtract from them)
        # Each neighbor contributes according to distribution_ratios
        neighbor_contributions = inflow_amount * distribution_ratios  # [batch, num_neighbors, length]

        # Subtract the contributions that go to neighbors
        # (this cell pulls from neighbors, so neighbors lose mass)
        for n, offset in enumerate(self.neighbor_offsets):
            neighbor_contribution = neighbor_contributions[:, n:n+1]
            # Shift to the neighbor's position (they lose this amount)
            shifted_contribution = torch.roll(neighbor_contribution, shifts=int(offset), dims=2)
            # This neighbor loses mass that goes to us
            next_field = next_field - shifted_contribution

        return next_field


if __name__ == "__main__":
    """Test FluxNet_U_1D for conservation and upper bound"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create model
    upper_bound = 1.0
    model = FluxNet_U_1D(
        in_channels=2,
        base_channels=32,
        num_blocks=4,
        kernel_size=3,
        neighborhood_size=15,
        upper_bound=upper_bound
    ).to(device)

    model.eval()

    # Test conservation and bounds
    print("\n=== Testing Conservation and Upper Bound ===")
    batch_size = 4
    length = 64

    # Create random input below upper bound
    conserved_field = torch.rand(batch_size, 1, length).to(device) * 0.5 + 0.2  # [0.2, 0.7]
    external_field = torch.randn(batch_size, 1, length).to(device)
    input_tensor = torch.cat([conserved_field, external_field], dim=1)

    # Compute initial mass
    initial_mass = conserved_field.sum(dim=2)

    # Forward pass
    with torch.no_grad():
        next_field, delta_field = model(input_tensor)

    # Compute final mass
    final_mass = next_field.sum(dim=2)

    # Check conservation
    mass_diff = (final_mass - initial_mass).abs()
    relative_error = mass_diff / (initial_mass + 1e-8)

    print(f"Initial mass: {initial_mass[0, 0].item():.6f}")
    print(f"Final mass: {final_mass[0, 0].item():.6f}")
    print(f"Max absolute mass difference: {mass_diff.max().item():.10f}")
    print(f"Max relative error: {relative_error.max().item():.10f}")

    # Check upper bound
    max_value = next_field.max().item()
    print(f"\nUpper bound constraint: <= {upper_bound}")
    print(f"Maximum value in output: {max_value:.10f}")

    if relative_error.max() < 1e-5:
        print("Conservation test PASSED")
    else:
        print("Conservation test FAILED")

    if max_value <= upper_bound + 1e-6:
        print("Upper bound test PASSED")
    else:
        print(f"Upper bound test FAILED (violation: {max_value - upper_bound:.10f})")

    # Test multi-step
    print("\n=== Testing Multi-Step ===")
    current = conserved_field.clone()
    initial_mass_total = current.sum()

    for step in range(10):
        input_tensor = torch.cat([current, external_field], dim=1)
        with torch.no_grad():
            current, _ = model(input_tensor)

        step_mass = current.sum()
        step_max = current.max()
        drift = (step_mass - initial_mass_total).abs() / (initial_mass_total + 1e-8)

        print(f"Step {step+1}: mass={step_mass.item():.6f}, drift={drift.item():.10f}, max={step_max.item():.6f}")

        if step_max > upper_bound + 1e-6:
            print(f"  Upper bound violated at step {step+1}!")

    print("\n=== Shape Test ===")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {next_field.shape}")
    print(f"Delta shape: {delta_field.shape}")

    print("\nAll tests completed!")
