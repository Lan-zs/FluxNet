"""
FluxNet-D 1D Dirichlet: Dual-Bounded Conservative Flux Network
with Dirichlet Boundary Conditions via Ghost Cell Method.

Input format:  [batch, in_channels, L+2]
    x[:, :, 0]     = left  Dirichlet boundary value
    x[:, :, 1:-1]  = physical interior domain (length L)
    x[:, :, -1]    = right Dirichlet boundary value

Output format: [batch, 1, L+2]
    output[:, :, 0]     = left  Dirichlet value (unchanged)
    output[:, :, 1:-1]  = predicted next state (length L)
    output[:, :, -1]    = right Dirichlet value (unchanged)

Conservation:
    On the ghost-extended domain (length L+2R), torch.roll is periodic,
    so total mass is EXACTLY preserved. Physical domain mass changes by
    the boundary flux only:
        M^{t+1}_phys = M^t_phys + Phi_boundary
    Flux-balance error (= extended-domain total change) is at machine
    precision (~1e-7 in float32).

Identity Channel (NEW):
    An extra channel is appended (as the last channel) to the backbone
    input, providing a spatial hint of ghost vs. interior positions.
    - "binary" mode:  ghost=0, interior=1
    - "smooth" mode:  ghost=0, sigmoid ramp over the first/last R
                      interior points, interior center=1
    This channel is generated internally; the external input interface
    is unchanged.  first_conv accepts (in_channels + 1) input channels
    to accommodate this extra channel.  _compute_transport() is
    completely unaffected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================
# Building blocks (ReplicatePad replaces CircularPad)
# ========================================================================

class ReplicatePad1D(nn.Module):
    """1D replicate padding for non-periodic boundary conditions."""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding), mode='replicate')


class DoubleConv1D(nn.Module):
    """Double convolution block with replicate padding."""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 act_fn=nn.ReLU, norm_1d=nn.BatchNorm1d):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            ReplicatePad1D(pad),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            norm_1d(out_channels),
            act_fn(),
            ReplicatePad1D(pad),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            norm_1d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)


# ========================================================================
# Main model
# ========================================================================

class FluxNet_D_Dirichlet_1D(nn.Module):
    """
    1D Dual-Bounded Flux Network with Dirichlet Boundary Conditions.

    Compared to the periodic version, the ONLY changes are:
      1. CircularPad  -> ReplicatePad  (backbone convolutions)
      2. forward()    : parse BCs -> ghost-extend -> backbone+transport
                        on extended domain -> crop physical -> reassemble
      3. _compute_transport() : COMPLETELY UNCHANGED

    Identity Channel (NEW — sole structural addition):
      An automatically-generated mask channel is appended to the backbone
      input, encoding ghost (0) vs. interior (1) positions.  Two modes:
        - "binary" : hard 0/1 mask
        - "smooth" : sigmoid ramp of width R inside the physical boundary
      first_conv therefore takes (in_channels + 1) input channels.
      Everything else — transport, bounds, output format — is untouched.

    Args:
        in_channels:       conserved field + external fields
        base_channels:     base feature width
        num_blocks:        number of residual blocks
        kernel_size:       convolution kernel size
        neighborhood_size: transport stencil width (odd, >= 3)
        lower_bound / upper_bound: physical bounds on the conserved field
        ghost_identity_mode:         "binary" or "smooth"
        ghost_identity_smooth_alpha: steepness of sigmoid ramp (smooth mode)
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
                 learnable_upper_bound=False,
                 ghost_identity_mode="binary",
                 ghost_identity_smooth_alpha=8.0):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"
        assert neighborhood_size >= 3, "neighborhood_size must be >= 3"
        if ghost_identity_mode not in ("binary", "smooth"):
            raise ValueError(
                f"Unknown ghost_identity_mode: '{ghost_identity_mode}'. "
                f"Must be 'binary' or 'smooth'.")

        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.radius = neighborhood_size // 2          # R — ghost width per side
        self.num_neighbors = neighborhood_size - 1    # K directional channels
        self.total_channels = 2 * (1 + self.num_neighbors)
        self.learnable_lower_bound = learnable_lower_bound
        self.learnable_upper_bound = learnable_upper_bound

        # ---- identity channel config (NEW) ----
        self.ghost_identity_mode = ghost_identity_mode
        self.ghost_identity_smooth_alpha = ghost_identity_smooth_alpha

        # ---- bounds ----
        if learnable_lower_bound:
            self.lower_bound_logit = nn.Parameter(
                self._inverse_sigmoid(lower_bound).data)
        else:
            self.register_buffer('lower_bound_value',
                                 torch.tensor(lower_bound).detach())

        if learnable_upper_bound:
            self.upper_bound_logit = nn.Parameter(
                self._inverse_sigmoid(upper_bound).data)
        else:
            self.register_buffer('upper_bound_value',
                                 torch.tensor(upper_bound).detach())

        # ---- backbone (ReplicatePad) ----
        # NOTE: first_conv takes (in_channels + 1) because we append one
        #       identity channel internally.  The user-facing in_channels
        #       is unchanged.
        self.first_conv = nn.Sequential(
            ReplicatePad1D(kernel_size // 2),
            nn.Conv1d(in_channels + 1, base_channels,
                      kernel_size=kernel_size, padding=0),
            norm_1d(base_channels),
            act_fn()
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv1D(base_channels, base_channels,
                             kernel_size, act_fn, norm_1d),
                nn.Conv1d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # ---- flux prediction head (1x1 conv, identical to periodic) ----
        self.flux_conv = nn.Conv1d(base_channels,
                                   self.total_channels, kernel_size=1)

        # ---- neighbor offsets (identical to periodic) ----
        offsets = [i for i in range(-self.radius, self.radius + 1) if i != 0]
        self.register_buffer('neighbor_offsets',
                             torch.tensor(offsets, dtype=torch.long))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _inverse_sigmoid(x, eps=1e-7):
        x = torch.clamp(torch.tensor(x), eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def lower_bound(self):
        if self.learnable_lower_bound:
            return torch.sigmoid(self.lower_bound_logit)
        return self.lower_bound_value

    @property
    def upper_bound(self):
        if self.learnable_upper_bound:
            return torch.sigmoid(self.upper_bound_logit)
        return self.upper_bound_value

    def _build_ghost_extended(self, x_phys, left_val, right_val):
        """
        Pad physical domain with R ghost cells on each side,
        filled with the corresponding Dirichlet value.

        Args
            x_phys   : [B, C, L]
            left_val : [B, C, 1]
            right_val: [B, C, 1]
        Returns
            x_ext    : [B, C, L + 2R]
        """
        R = self.radius
        left_ghost = left_val.expand(-1, -1, R)     # [B, C, R]
        right_ghost = right_val.expand(-1, -1, R)   # [B, C, R]
        return torch.cat([left_ghost, x_phys, right_ghost], dim=2)

    def _build_identity_channel(self, batch_size, L, device, dtype):
        """
        Construct the ghost/interior identity channel on the
        ghost-extended domain.

        Args
            batch_size : int
            L          : int, physical interior length
            device     : torch.device
            dtype      : torch.dtype

        Returns
            identity   : [B, 1, L + 2R]

        Modes
        -----
        "binary":
            ghost positions = 0, interior positions = 1.

        "smooth":
            ghost positions = 0.
            Interior positions near the boundary (within distance R
            of the ghost region) ramp smoothly from >0 up to 1 via a
            normalized sigmoid.  Interior positions at distance >= R+1
            from the boundary are exactly 1.  The ramp is symmetric
            on both sides.

            Precisely, for an interior index i (0-based):
                d = min(i + 1, L - i)          # 1-based distance to nearest edge
                t = d / (R + 1)                # ∈ (0, 1] for d ∈ [1, R+1]
                raw   = σ(α · (t − 0.5))
                s0    = σ(α · (0.0 − 0.5))    # reference for ghost boundary
                s1    = σ(α · (1.0 − 0.5))    # reference for d = R+1
                value = clamp((raw − s0) / (s1 − s0), 0, 1)
            For d >= R+1, value is set to 1 directly.
        """
        R = self.radius
        L_ext = L + 2 * R

        if self.ghost_identity_mode == "binary":
            # ---- binary: ghost=0, interior=1 ----
            identity = torch.zeros(1, 1, L_ext, device=device, dtype=dtype)
            identity[:, :, R:R + L] = 1.0

        elif self.ghost_identity_mode == "smooth":
            # ---- smooth: sigmoid ramp over first/last R interior cells ----
            alpha = self.ghost_identity_smooth_alpha

            # distance of each interior position to the nearest boundary
            i = torch.arange(L, device=device, dtype=dtype)   # [L]
            d_left = i + 1.0          # 1-based
            d_right = float(L) - i    # 1-based from right
            d = torch.minimum(d_left, d_right)                # [L]

            # normalised position for sigmoid
            t = d / (R + 1)                                   # (0, …]

            # sigmoid references
            s0 = torch.sigmoid(torch.tensor(alpha * (0.0 - 0.5),
                                            device=device, dtype=dtype))
            s1 = torch.sigmoid(torch.tensor(alpha * (1.0 - 0.5),
                                            device=device, dtype=dtype))

            raw = torch.sigmoid(alpha * (t - 0.5))
            value = (raw - s0) / (s1 - s0)
            value = value.clamp(0.0, 1.0)

            # positions deep inside (d >= R+1) → exactly 1
            value = torch.where(d >= (R + 1), torch.ones_like(value), value)

            # assemble extended domain
            identity = torch.zeros(1, 1, L_ext, device=device, dtype=dtype)
            identity[:, :, R:R + L] = value.unsqueeze(0).unsqueeze(0)

        else:
            # Should not reach here (validated in __init__), but be safe.
            raise ValueError(
                f"Unknown ghost_identity_mode: '{self.ghost_identity_mode}'. "
                f"Must be 'binary' or 'smooth'.")

        # expand to batch (memory-efficient view)
        return identity.expand(batch_size, -1, -1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args
            x: [B, in_channels, L+2]
               position  0   = left  Dirichlet BC
               positions 1…L = physical interior
               position  L+1 = right Dirichlet BC

        Returns
            next_full          : [B, 1, L+2]  next state with BCs at ends
            outflow_change     : [B, 1, L]    outflow-branch Δu (physical)
            inflow_change      : [B, 1, L]    inflow-branch  Δu (physical)
            boundary_flux      : [B, 1]       net mass entering physical domain
            flux_balance_error : [B, 1]       extended-domain conservation error
                                              (should be ~1e-7 in float32)
        """
        R = self.radius

        # =========== 1. Parse input ===========
        left_bc_all  = x[:, :, 0:1]      # [B, C_in, 1]
        right_bc_all = x[:, :, -1:]       # [B, C_in, 1]
        x_phys       = x[:, :, 1:-1]     # [B, C_in, L]

        # =========== 2. Ghost-extend ALL channels for backbone ===========
        x_ext = self._build_ghost_extended(
            x_phys, left_bc_all, right_bc_all)        # [B, C_in, L+2R]

        # =========== 2b. Append identity channel (NEW) ===========
        #   The identity channel is the LAST channel of the backbone input.
        #   It encodes ghost (0) vs. interior (1/smooth) positions.
        #   This is the ONLY structural change compared to the original model.
        B = x.shape[0]
        L = x_phys.shape[2]
        identity_ext = self._build_identity_channel(
            B, L, x.device, x.dtype)                  # [B, 1, L+2R]
        x_ext = torch.cat([x_ext, identity_ext], dim=1)  # [B, C_in+1, L+2R]

        # =========== 3. Backbone on extended domain ===========
        features = self.first_conv(x_ext)             # [B, base_ch, L+2R]
        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # =========== 4. Flux parameters on extended domain ===========
        raw = self.flux_conv(features)                # [B, total_ch, L+2R]

        K = self.num_neighbors
        outflow_pct  = torch.sigmoid(raw[:, 0:1])
        outflow_dist = F.softmax(raw[:, 1:K+1], dim=1)
        inflow_pct   = torch.sigmoid(raw[:, K+1:K+2])
        inflow_dist  = F.softmax(raw[:, K+2:], dim=1)

        # =========== 5. Conserved field on extended domain ===========
        conserved_phys = x_phys[:, 0:1]               # [B, 1, L]
        left_bc_cons   = x[:, 0:1, 0:1]               # [B, 1, 1]
        right_bc_cons  = x[:, 0:1, -1:]               # [B, 1, 1]
        conserved_ext  = self._build_ghost_extended(
            conserved_phys, left_bc_cons, right_bc_cons)  # [B, 1, L+2R]

        # =========== 6. Transport — COMPLETELY UNCHANGED ===========
        out_ext, in_ext = self._compute_transport(
            conserved_ext, outflow_pct, outflow_dist,
            inflow_pct, inflow_dist)

        # =========== 7. Combined change on extended domain ===========
        combined_ext = (out_ext + in_ext) / 2         # [B, 1, L+2R]

        # =========== 8. Crop physical domain ===========
        outflow_change = out_ext[:, :, R:-R]          # [B, 1, L]
        inflow_change  = in_ext[:, :, R:-R]           # [B, 1, L]
        combined_phys  = combined_ext[:, :, R:-R]     # [B, 1, L]

        next_phys = conserved_phys + combined_phys    # [B, 1, L]

        # =========== 9. Diagnostics ===========
        # Net boundary flux = physical mass change
        boundary_flux = combined_phys.sum(dim=2)      # [B, 1]
        # Extended-domain total change (should be ~0)
        flux_balance_error = combined_ext.sum(dim=2)  # [B, 1]

        # =========== 10. Reassemble with BCs ===========
        next_full = torch.cat(
            [left_bc_cons, next_phys, right_bc_cons], dim=2)  # [B, 1, L+2]

        return next_full, outflow_change, inflow_change

    # ------------------------------------------------------------------
    # transport kernel — IDENTICAL to periodic version
    # ------------------------------------------------------------------
    def _compute_transport(self, current_field,
                           outflow_pct, outflow_dist,
                           inflow_pct, inflow_dist):
        """
        Conservative transport via directional fluxes.
        Operates on whatever spatial length is given.
        *** This function is BYTE-FOR-BYTE identical to the periodic code. ***
        """
        # --- Outflow branch (lower-bound capacity) ---
        available_outflow = current_field - self.lower_bound
        outflow_amount = available_outflow * outflow_pct
        outflow_change = -outflow_amount
        outflow_to_neighbors = outflow_amount * outflow_dist

        # --- Inflow branch (upper-bound capacity) ---
        available_inflow = self.upper_bound - current_field
        inflow_amount = available_inflow * inflow_pct
        inflow_change = inflow_amount
        inflow_from_neighbors = inflow_amount * inflow_dist

        # --- Shift and accumulate ---
        for n, offset in enumerate(self.neighbor_offsets):
            shifted_out = torch.roll(
                outflow_to_neighbors[:, n:n+1],
                shifts=-int(offset), dims=2)
            outflow_change = outflow_change + shifted_out

            shifted_in = torch.roll(
                inflow_from_neighbors[:, n:n+1],
                shifts=int(offset), dims=2)
            inflow_change = inflow_change - shifted_in

        return outflow_change, inflow_change

