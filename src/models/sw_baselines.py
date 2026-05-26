"""
Shallow Water Equation Baselines with Projection

Baseline models (FNO, CNN) enhanced with post-hoc projection to ensure:
1. Box constraint: h >= 0 (water depth must be non-negative)
2. Mass conservation: sum(h) is preserved

Two projection modes:
- 'box': Only clamp h to be >= 0 (no mass conservation)
- 'box_mass': First clamp h >= 0, then scale to preserve total mass

These serve as strong baselines to compare against FluxNet-LAP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Projection functions
# ============================================================================

def box_projection(h, lower_bound=0.0):
    """
    Simply clamp h to be >= lower_bound

    Args:
        h: [batch, 1, H, W] water depth
        lower_bound: minimum allowed value (default 0.0)

    Returns:
        h_projected: h with values >= lower_bound
    """
    return torch.clamp(h, min=lower_bound)


def box_mass_projection(h, h_original, lower_bound=0.0, eps=1e-8):
    """
    Two-step projection:
    1. Clamp h >= lower_bound (box constraint)
    2. Scale to preserve original total mass

    Note: After clamping, total mass might increase (if there were negative values).
    We scale down proportionally from the lower_bound to preserve mass.

    Args:
        h: [batch, 1, H, W] predicted water depth
        h_original: [batch, 1, H, W] original water depth (for mass reference)
        lower_bound: minimum allowed value (default 0.0)
        eps: small value for numerical stability

    Returns:
        h_projected: h satisfying both box and mass constraints
    """
    # Step 1: Box projection
    h_clamped = torch.clamp(h, min=lower_bound)

    # Step 2: Compute mass difference
    original_mass = h_original.sum(dim=(2, 3), keepdim=True)  # [batch, 1, 1, 1]
    clamped_mass = h_clamped.sum(dim=(2, 3), keepdim=True)

    # Calculate available mass to redistribute (mass above lower_bound)
    h_above = h_clamped - lower_bound  # All values >= 0
    available_mass = h_above.sum(dim=(2, 3), keepdim=True)

    # Scale factor to match original mass
    # new_mass = lower_bound * N + scale * available_mass = original_mass
    # scale = (original_mass - lower_bound * N) / available_mass
    N = h.shape[2] * h.shape[3]  # total number of grid points
    target_available = original_mass - lower_bound * N

    # Ensure target_available >= 0 (original mass should at least cover the base)
    target_available = torch.clamp(target_available, min=0)

    # Compute scale factor
    scale = target_available / (available_mass + eps)
    scale = torch.clamp(scale, min=0, max=2.0)  # Limit scaling to avoid extreme values

    # Apply scaling
    h_projected = lower_bound + h_above * scale

    return h_projected


# ============================================================================
# FNO for Shallow Water with Projection
# ============================================================================

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """Single FNO Block"""

    def __init__(self, channels, modes1, modes2, act_fn=nn.GELU):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = act_fn()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        return self.act(x1 + x2)


class FNO_SW_Proj(nn.Module):
    """
    FNO for Shallow Water with Projection

    After network prediction, applies projection to ensure:
    - h >= 0 (box constraint)
    - Optional: mass conservation

    Args:
        modes1, modes2: Number of Fourier modes
        width: Hidden channel width
        num_layers: Number of FNO blocks
        projection_mode: 'none', 'box', or 'box_mass'
        prediction_mode: 'direct' or 'residual'
    """

    def __init__(self,
                 modes1=16,
                 modes2=16,
                 width=64,
                 num_layers=4,
                 projection_mode='box_mass',
                 prediction_mode='residual'):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.projection_mode = projection_mode
        self.prediction_mode = prediction_mode

        in_channels = 3  # h, mx, my
        out_channels = 3

        # Lift to higher dimension
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2)
            for _ in range(num_layers)
        ])

        # Project back to output dimension
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: [batch, 3, H, W] with projected h field
        """
        h_input = x[:, 0:1]  # Original h for mass reference

        # Network forward
        x_lifted = self.lift(x)
        for block in self.fno_blocks:
            x_lifted = block(x_lifted)
        output = self.project(x_lifted)

        # Prediction mode
        if self.prediction_mode == 'residual':
            next_state = x + output
        else:
            next_state = output

        # Extract fields
        h_pred = next_state[:, 0:1]
        mx_pred = next_state[:, 1:2]
        my_pred = next_state[:, 2:3]

        # Apply projection to h
        if self.projection_mode == 'box':
            h_proj = box_projection(h_pred, lower_bound=0.0)
        elif self.projection_mode == 'box_mass':
            h_proj = box_mass_projection(h_pred, h_input, lower_bound=0.0)
        else:
            h_proj = h_pred

        # Reconstruct output
        next_state = torch.cat([h_proj, mx_pred, my_pred], dim=1)

        return (next_state,)


# ============================================================================
# CNN for Shallow Water with Projection
# ============================================================================

class CircularPad2D(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.GELU, norm_2d=nn.BatchNorm2d):
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


class CNN_SW_Proj(nn.Module):
    """
    CNN for Shallow Water with Projection

    Standard CNN architecture with post-hoc projection for:
    - h >= 0 (box constraint)
    - Optional: mass conservation

    Args:
        base_channels: Base feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolutional kernel size
        projection_mode: 'none', 'box', or 'box_mass'
        prediction_mode: 'direct' or 'residual'
    """

    def __init__(self,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 projection_mode='box_mass',
                 prediction_mode='residual'):
        super().__init__()

        self.projection_mode = projection_mode
        self.prediction_mode = prediction_mode

        in_channels = 3  # h, mx, my
        out_channels = 3

        # First conv
        self.first_conv = nn.Sequential(
            CircularPad2D(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size, padding=0),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv2D(base_channels, base_channels, kernel_size),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Output layer
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [batch, 3, H, W] containing [h, mx, my]

        Returns:
            next_state: [batch, 3, H, W] with projected h field
        """
        h_input = x[:, 0:1]  # Original h for mass reference

        # Network forward
        features = self.first_conv(x)

        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        output = self.output_conv(features)

        # Prediction mode
        if self.prediction_mode == 'residual':
            next_state = x + output
        else:
            next_state = output

        # Extract fields
        h_pred = next_state[:, 0:1]
        mx_pred = next_state[:, 1:2]
        my_pred = next_state[:, 2:3]

        # Apply projection to h
        if self.projection_mode == 'box':
            h_proj = box_projection(h_pred, lower_bound=0.0)
        elif self.projection_mode == 'box_mass':
            h_proj = box_mass_projection(h_pred, h_input, lower_bound=0.0)
        else:
            h_proj = h_pred

        # Reconstruct output
        next_state = torch.cat([h_proj, mx_pred, my_pred], dim=1)

        return (next_state,)


# ============================================================================
# FluxNet Baseline for Shallow Water (direct prediction, no flux structure)
# ============================================================================

class FluxNet_SW_Baseline(nn.Module):
    """
    Direct prediction baseline using FluxNet-style architecture
    but without the flux-based update structure.

    This is essentially a CNN that directly predicts [h, mx, my]
    with optional projection.

    Args:
        base_channels: Base feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolutional kernel size
        projection_mode: 'none', 'box', or 'box_mass'
    """

    def __init__(self,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 projection_mode='box'):
        super().__init__()

        self.projection_mode = projection_mode

        # Use the CNN_SW_Proj with residual prediction
        self.backbone = CNN_SW_Proj(
            base_channels=base_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            projection_mode=projection_mode,
            prediction_mode='residual'
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Shallow Water Baselines on: {device}")

    batch_size = 4
    H, W = 64, 64

    # Create input
    h = torch.rand(batch_size, 1, H, W).to(device) + 0.5
    mx = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    my = torch.randn(batch_size, 1, H, W).to(device) * 0.1
    input_state = torch.cat([h, mx, my], dim=1)

    # Test FNO_SW_Proj
    print("\n=== Testing FNO_SW_Proj (box_mass projection) ===")
    model_fno = FNO_SW_Proj(
        modes1=16, modes2=16, width=32, num_layers=4,
        projection_mode='box_mass', prediction_mode='residual'
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model_fno.parameters()):,}")

    model_fno.eval()
    with torch.no_grad():
        output, = model_fno(input_state)

    h_in = input_state[:, 0]
    h_out = output[:, 0]

    print(f"h input range: [{h_in.min().item():.4f}, {h_in.max().item():.4f}]")
    print(f"h output range: [{h_out.min().item():.4f}, {h_out.max().item():.4f}]")
    print(f"h >= 0: {(h_out >= 0).all().item()}")

    mass_in = h_in.sum().item()
    mass_out = h_out.sum().item()
    mass_drift = abs(mass_out - mass_in) / (abs(mass_in) + 1e-8)
    print(f"Mass conservation drift: {mass_drift:.6e}")

    # Test CNN_SW_Proj
    print("\n=== Testing CNN_SW_Proj (box_mass projection) ===")
    model_cnn = CNN_SW_Proj(
        base_channels=32, num_blocks=4, kernel_size=3,
        projection_mode='box_mass', prediction_mode='residual'
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

    model_cnn.eval()
    with torch.no_grad():
        output_cnn, = model_cnn(input_state)

    h_out_cnn = output_cnn[:, 0]
    print(f"h output range: [{h_out_cnn.min().item():.4f}, {h_out_cnn.max().item():.4f}]")
    print(f"h >= 0: {(h_out_cnn >= 0).all().item()}")

    mass_out_cnn = h_out_cnn.sum().item()
    mass_drift_cnn = abs(mass_out_cnn - mass_in) / (abs(mass_in) + 1e-8)
    print(f"Mass conservation drift: {mass_drift_cnn:.6e}")

    # Test box-only projection
    print("\n=== Testing FNO_SW_Proj (box-only projection) ===")
    model_box = FNO_SW_Proj(
        modes1=16, modes2=16, width=32, num_layers=4,
        projection_mode='box', prediction_mode='residual'
    ).to(device)

    model_box.eval()
    with torch.no_grad():
        output_box, = model_box(input_state)

    h_out_box = output_box[:, 0]
    print(f"h >= 0: {(h_out_box >= 0).all().item()}")
    mass_out_box = h_out_box.sum().item()
    mass_drift_box = abs(mass_out_box - mass_in) / (abs(mass_in) + 1e-8)
    print(f"Mass conservation drift (no mass proj): {mass_drift_box:.6e}")

    print("\nShallow Water Baselines OK!")
