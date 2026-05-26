"""
FluxNet-L 1D: Lower-Bounded Conservative Flux Network for 1D problems

This model:
- Guarantees mass conservation through flux-based updates
- Enforces lower bound constraint (e.g., concentration >= 0)
- Suitable for lower-bounded conserved quantities (e.g., concentration, water depth)
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


class FluxNet_L_1D(nn.Module):
    """
    1D Lower-Bounded Flux Network

    Args:
        in_channels: Number of input channels (conserved field + external fields)
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for convolutions
        act_fn: Activation function
        norm_1d: Normalization layer
        neighborhood_size: Size of the neighborhood stencil (must be odd)
        lower_bound: Lower bound for the conserved field
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_1d=nn.BatchNorm1d,
                 neighborhood_size=15,
                 lower_bound=0.0):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"

        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.register_buffer('lower_bound_value', torch.tensor(lower_bound))

        # Number of neighbors (excluding center point)
        self.num_neighbors = neighborhood_size - 1

        # Total channels: 1 for outflow percentage + num_neighbors for distribution
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
            if i != 0:  # Exclude center point
                neighbor_offsets.append(i)
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    @property
    def lower_bound(self):
        """Get current lower bound value"""
        return self.lower_bound_value

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

        # Split into outflow percentage and distribution ratios
        outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])  # [batch, 1, length]
        distribution_logits = raw_fluxes[:, 1:]  # [batch, num_neighbors, length]
        distribution_ratios = F.softmax(distribution_logits, dim=1)

        # Get conserved field
        conserved_field = x[:, 0:1]  # [batch, 1, length]

        # Compute transport with lower bound constraint
        next_field = self._compute_transport(
            conserved_field,
            outflow_percentage,
            distribution_ratios
        )

        return next_field, next_field - conserved_field

    def _compute_transport(self, current_field, outflow_percentage, distribution_ratios):
        """
        Compute conservative transport with lower bound constraint

        The key idea: only the amount above the lower bound can flow out

        Args:
            current_field: [batch, 1, length]
            outflow_percentage: [batch, 1, length] - fraction of available amount to send out
            distribution_ratios: [batch, num_neighbors, length] - how to distribute outflow

        Returns:
            next_field: [batch, 1, length], guaranteed to be >= lower_bound
        """
        # Amount available for outflow (above lower bound)
        available = current_field - self.lower_bound

        # Total outflow amount
        outflow_amount = available * outflow_percentage

        # Start with current field minus outflow
        next_field = current_field - outflow_amount

        # Distribute outflow to neighbors
        neighbor_flows = outflow_amount * distribution_ratios  # [batch, num_neighbors, length]

        # Add incoming flows from neighbors
        for n, offset in enumerate(self.neighbor_offsets):
            neighbor_flow = neighbor_flows[:, n:n+1]
            # Shift in opposite direction to get incoming contribution
            shifted_flow = torch.roll(neighbor_flow, shifts=-int(offset), dims=2)
            next_field = next_field + shifted_flow

        return next_field


if __name__ == "__main__":
    """Test FluxNet_L_1D for conservation and bounds"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create model
    lower_bound = 0.0
    model = FluxNet_L_1D(
        in_channels=2,
        base_channels=32,
        num_blocks=4,
        kernel_size=3,
        neighborhood_size=15,
        lower_bound=lower_bound
    ).to(device)

    model.eval()

    # Test conservation and bounds
    print("\n=== Testing Conservation and Lower Bound ===")
    batch_size = 4
    length = 64

    # Create random input above lower bound
    conserved_field = torch.rand(batch_size, 1, length).to(device) * 0.5 + 0.5  # [0.5, 1.0]
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

    # Check lower bound
    min_value = next_field.min().item()
    print(f"\nLower bound constraint: >= {lower_bound}")
    print(f"Minimum value in output: {min_value:.10f}")

    if relative_error.max() < 1e-5:
        print("✓ Conservation test PASSED")
    else:
        print("✗ Conservation test FAILED")

    if min_value >= lower_bound - 1e-6:
        print("✓ Lower bound test PASSED")
    else:
        print(f"✗ Lower bound test FAILED (violation: {lower_bound - min_value:.10f})")

    # Test multi-step conservation and bounds
    print("\n=== Testing Multi-Step Conservation and Bounds ===")
    current = conserved_field.clone()
    initial_mass_total = current.sum()

    for step in range(10):
        input_tensor = torch.cat([current, external_field], dim=1)
        with torch.no_grad():
            current, _ = model(input_tensor)

        step_mass = current.sum()
        step_min = current.min()
        drift = (step_mass - initial_mass_total).abs() / (initial_mass_total + 1e-8)

        print(f"Step {step+1}: mass={step_mass.item():.6f}, drift={drift.item():.10f}, min={step_min.item():.6f}")

        if step_min < lower_bound - 1e-6:
            print(f"  ✗ Lower bound violated at step {step+1}!")

    print("\n=== Shape Test ===")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {next_field.shape}")
    print(f"Delta shape: {delta_field.shape}")

    print("\nAll tests completed!")
