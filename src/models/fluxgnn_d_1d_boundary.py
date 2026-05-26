"""
fluxgnn_fluxnetd_traffic_1d.py

Hybrid model: FluxGNN backbone + FluxNet-D transport head.

Input:  [B, in_channels, 258]
    x[:, :, 0]       = left  Dirichlet BC (ρ_L + aux)
    x[:, :, 1:257]   = 256 internal FV cells
    x[:, :, 257]     = right Dirichlet BC (ρ_R + aux)

Output: next_full, outflow_change, inflow_change
    next_full:       [B, 1, 258]  with BCs preserved at positions 0 and 257
    outflow_change:  [B, 1, 256]  outflow-branch Δρ on physical domain
    inflow_change:   [B, 1, 256]  inflow-branch  Δρ on physical domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# FluxGNN Backbone Components (from fluxgnn_traffic_1d.py)
# ============================================================================

class MessagePassingLayer(nn.Module):
    """One round of 1D face-based message passing.

    1. Build face features by concatenating left/right cell hidden states.
       Boundary faces use the encoded BC vector.
    2. Face network produces per-face messages.
    3. Aggregate: cell_j receives face_msg[j] (left) + face_msg[j+1] (right).
    4. Residual cell update.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Face network: [B, 2H, 257] → [B, H, 257]
        self.face_net = nn.Sequential(
            nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        # Cell update: [B, H, 256] → [B, H, 256]
        self.cell_update = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )

    def forward(self, h, h_bc_left, h_bc_right):
        """
        Args:
            h:          [B, H, 256]  current cell hidden states
            h_bc_left:  [B, H, 1]   encoded left  Dirichlet BC
            h_bc_right: [B, H, 1]   encoded right Dirichlet BC
        Returns:
            h_updated:  [B, H, 256]  updated cell hidden states
        """
        # Build 257 face features: face j = (left_state[j], right_state[j])
        #   face 0:      left = BC_L,      right = cell[0]
        #   face 1..255: left = cell[j-1],  right = cell[j]
        #   face 256:    left = cell[255],  right = BC_R
        left_states  = torch.cat([h_bc_left,  h], dim=-1)   # [B, H, 257]
        right_states = torch.cat([h, h_bc_right], dim=-1)   # [B, H, 257]
        face_input   = torch.cat([left_states, right_states], dim=1)  # [B, 2H, 257]

        face_msgs = self.face_net(face_input)  # [B, H, 257]

        # Aggregate to cells: cell j ← msg from left face j + right face j+1
        cell_agg = face_msgs[:, :, :-1] + face_msgs[:, :, 1:]  # [B, H, 256]

        # Residual update
        h_updated = h + self.cell_update(cell_agg)  # [B, H, 256]
        return h_updated


# ============================================================================
# Main Hybrid Model
# ============================================================================

class FluxGNN_D_1D(nn.Module):
    """
    Hybrid: FluxGNN backbone (message-passing) + FluxNet-D transport head
    (dual-bounded conservative transport with configurable neighborhood).

    This model verifies whether replacing FluxNet-D's CNN backbone with
    FluxGNN's face-based message-passing backbone improves performance,
    while keeping the transport head (the key innovation of FluxNet-D)
    completely unchanged.

    Args:
        in_channels:       input channels (ch0 = density, rest = auxiliary)
        out_channels:      always 1 for the transport head (density only)
        hidden_dim:        FluxGNN backbone hidden width
        num_mp_layers:     number of message-passing rounds in backbone
        neighborhood_size: transport stencil width (odd, >= 3)
        lower_bound:       physical lower bound on density
        upper_bound:       physical upper bound on density
        ghost_identity_mode: "binary" or "smooth"
        ghost_identity_smooth_alpha: steepness for smooth mode sigmoid ramp
        prediction_mode:   accepted for API compat, not used
        bound_mode:        accepted for API compat, not used
    """

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 # --- FluxGNN backbone params ---
                 hidden_dim: int = 32,
                 num_mp_layers: int = 4,
                 # --- FluxNet-D head params ---
                 neighborhood_size: int = 15,
                 lower_bound: float = 0.0,
                 upper_bound: float = 1.0,
                 ghost_identity_mode: str = "binary",
                 ghost_identity_smooth_alpha: float = 8.0,
                 # --- API compat (unused) ---
                 prediction_mode: str = 'direct',
                 bound_mode: str = 'none',
                 # --- swallow FNO-specific kwargs ---
                 **kwargs):
        super().__init__()

        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"
        assert neighborhood_size >= 3, "neighborhood_size must be >= 3"
        if ghost_identity_mode not in ("binary", "smooth"):
            raise ValueError(
                f"Unknown ghost_identity_mode: '{ghost_identity_mode}'. "
                f"Must be 'binary' or 'smooth'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # Transport stencil params
        self.neighborhood_size = neighborhood_size
        self.radius = neighborhood_size // 2             # R ghost cells per side
        self.num_neighbors = neighborhood_size - 1       # K directional channels
        self.total_channels = 2 * (1 + self.num_neighbors)

        # Identity channel config
        self.ghost_identity_mode = ghost_identity_mode
        self.ghost_identity_smooth_alpha = ghost_identity_smooth_alpha

        # API compat storage
        self.prediction_mode = prediction_mode
        self.bound_mode = bound_mode

        # Bounds
        self.register_buffer('lower_bound_value',
                             torch.tensor(lower_bound, dtype=torch.float32))
        self.register_buffer('upper_bound_value',
                             torch.tensor(upper_bound, dtype=torch.float32))

        # === FluxGNN Backbone ================================================
        # Cell encoder: [B, C_in, 256] → [B, H, 256]
        self.cell_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        # BC encoder: [B, C_in, 1] → [B, H, 1]  (shared for left & right)
        self.bc_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_mp_layers)
        ])

        # === FluxNet-D Transport Head ========================================
        # Input: ghost-extended hidden + identity channel → [B, H+1, L+2R]
        # Output: transport parameters → [B, total_channels, L+2R]
        self.flux_conv = nn.Conv1d(
            hidden_dim + 1,          # +1 for identity channel
            self.total_channels,
            kernel_size=1,
        )

        # Neighbor offsets for torch.roll transport
        offsets = [i for i in range(-self.radius, self.radius + 1) if i != 0]
        self.register_buffer('neighbor_offsets',
                             torch.tensor(offsets, dtype=torch.long))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lower_bound(self):
        return self.lower_bound_value

    @property
    def upper_bound(self):
        return self.upper_bound_value

    # ------------------------------------------------------------------
    # Ghost extension helpers (from FluxNet-D, adapted for hidden states)
    # ------------------------------------------------------------------

    def _build_ghost_extended(self, x_phys, left_val, right_val):
        """Pad physical domain with R ghost cells on each side.

        Args:
            x_phys:    [B, C, L]
            left_val:  [B, C, 1]
            right_val: [B, C, 1]
        Returns:
            x_ext:     [B, C, L + 2R]
        """
        R = self.radius
        left_ghost  = left_val.expand(-1, -1, R)   # [B, C, R]
        right_ghost = right_val.expand(-1, -1, R)   # [B, C, R]
        return torch.cat([left_ghost, x_phys, right_ghost], dim=2)

    def _build_identity_channel(self, batch_size, L, device, dtype):
        """Construct ghost/interior identity channel on extended domain.

        Marks ghost positions as 0 and interior positions as 1 (binary)
        or with a smooth sigmoid ramp (smooth mode).

        Args:
            batch_size: int
            L:          int, physical interior length
            device:     torch.device
            dtype:      torch.dtype
        Returns:
            identity:   [B, 1, L + 2R]
        """
        R = self.radius
        L_ext = L + 2 * R

        if self.ghost_identity_mode == "binary":
            identity = torch.zeros(1, 1, L_ext, device=device, dtype=dtype)
            identity[:, :, R:R + L] = 1.0

        elif self.ghost_identity_mode == "smooth":
            alpha = self.ghost_identity_smooth_alpha
            i = torch.arange(L, device=device, dtype=dtype)
            d = torch.minimum(i + 1.0, float(L) - i)
            t = d / (R + 1)
            s0 = torch.sigmoid(torch.tensor(
                alpha * (0.0 - 0.5), device=device, dtype=dtype))
            s1 = torch.sigmoid(torch.tensor(
                alpha * (1.0 - 0.5), device=device, dtype=dtype))
            raw = torch.sigmoid(alpha * (t - 0.5))
            value = ((raw - s0) / (s1 - s0)).clamp(0.0, 1.0)
            value = torch.where(d >= (R + 1), torch.ones_like(value), value)
            identity = torch.zeros(1, 1, L_ext, device=device, dtype=dtype)
            identity[:, :, R:R + L] = value.unsqueeze(0).unsqueeze(0)

        return identity.expand(batch_size, -1, -1)  # [B, 1, L+2R]

    # ------------------------------------------------------------------
    # Transport kernel (IDENTICAL to FluxNet-D, byte-for-byte)
    # ------------------------------------------------------------------

    def _compute_transport(self, current_field,
                           outflow_pct, outflow_dist,
                           inflow_pct, inflow_dist):
        """
        Conservative transport via directional fluxes on extended domain.
        Uses torch.roll (periodic on extended domain) for exact conservation.

        Args:
            current_field: [B, 1, L_ext]   conserved field on extended domain
            outflow_pct:   [B, 1, L_ext]   sigmoid outflow fraction
            outflow_dist:  [B, K, L_ext]   softmax directional distribution
            inflow_pct:    [B, 1, L_ext]   sigmoid inflow fraction
            inflow_dist:   [B, K, L_ext]   softmax directional distribution

        Returns:
            outflow_change: [B, 1, L_ext]  outflow-branch Δρ
            inflow_change:  [B, 1, L_ext]  inflow-branch  Δρ
        """
        # --- Outflow branch (capacity above lower_bound) ---
        available_outflow = current_field - self.lower_bound   # [B, 1, L_ext]
        outflow_amount = available_outflow * outflow_pct       # [B, 1, L_ext]
        outflow_change = -outflow_amount                       # this cell loses
        outflow_to_neighbors = outflow_amount * outflow_dist   # [B, K, L_ext]

        # --- Inflow branch (capacity below upper_bound) ---
        available_inflow = self.upper_bound - current_field    # [B, 1, L_ext]
        inflow_amount = available_inflow * inflow_pct          # [B, 1, L_ext]
        inflow_change = inflow_amount                          # this cell gains
        inflow_from_neighbors = inflow_amount * inflow_dist    # [B, K, L_ext]

        # --- Shift and accumulate ---
        for n, offset in enumerate(self.neighbor_offsets):
            # Outflow from cell at (pos+offset) arrives at current pos
            shifted_out = torch.roll(
                outflow_to_neighbors[:, n:n+1],
                shifts=-int(offset), dims=2)
            outflow_change = outflow_change + shifted_out

            # Inflow demand from cell at (pos-offset) draws from current pos
            shifted_in = torch.roll(
                inflow_from_neighbors[:, n:n+1],
                shifts=int(offset), dims=2)
            inflow_change = inflow_change - shifted_in

        return outflow_change, inflow_change

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, 258]
               x[:, :, 0]       – left  Dirichlet BC (ρ_L + aux features)
               x[:, :, 1:257]   – 256 internal FV cells
               x[:, :, 257]     – right Dirichlet BC (ρ_R + aux features)

        Returns:
            next_full:       [B, 1, 258]  predicted next state with BCs
            outflow_change:  [B, 1, 256]  outflow-branch Δρ (physical domain)
            inflow_change:   [B, 1, 256]  inflow-branch  Δρ (physical domain)
        """
        R = self.radius
        B = x.shape[0]
        L = 256  # number of interior cells

        # =====================================================================
        # 1. Parse input
        # =====================================================================
        bc_left   = x[:, :, 0:1]       # [B, C_in, 1]
        interior  = x[:, :, 1:257]     # [B, C_in, 256]
        bc_right  = x[:, :, 257:258]   # [B, C_in, 1]

        # =====================================================================
        # 2. FluxGNN Backbone: encode + message passing
        # =====================================================================
        h      = self.cell_encoder(interior)   # [B, H, 256]
        h_bc_L = self.bc_encoder(bc_left)      # [B, H, 1]
        h_bc_R = self.bc_encoder(bc_right)     # [B, H, 1]

        for mp in self.mp_layers:
            h = mp(h, h_bc_L, h_bc_R)         # [B, H, 256]

        # =====================================================================
        # 3. Bridge: ghost-extend hidden states for transport head
        #    Ghost cells = BC hidden states replicated R times per side
        # =====================================================================
        h_ext = self._build_ghost_extended(
            h, h_bc_L, h_bc_R)                 # [B, H, L+2R]

        # Append identity channel to mark ghost (0) vs interior (1)
        identity_ch = self._build_identity_channel(
            B, L, x.device, x.dtype)           # [B, 1, L+2R]
        h_ext_with_id = torch.cat(
            [h_ext, identity_ch], dim=1)       # [B, H+1, L+2R]

        # =====================================================================
        # 4. FluxNet-D Transport Head: predict transport parameters
        # =====================================================================
        raw = self.flux_conv(h_ext_with_id)    # [B, total_ch, L+2R]

        K = self.num_neighbors
        outflow_pct  = torch.sigmoid(raw[:, 0:1])        # [B, 1, L+2R]
        outflow_dist = F.softmax(raw[:, 1:K+1], dim=1)   # [B, K, L+2R]
        inflow_pct   = torch.sigmoid(raw[:, K+1:K+2])    # [B, 1, L+2R]
        inflow_dist  = F.softmax(raw[:, K+2:], dim=1)    # [B, K, L+2R]

        # =====================================================================
        # 5. Build ghost-extended conserved field (channel 0 = density)
        # =====================================================================
        conserved_phys  = interior[:, 0:1, :]  # [B, 1, 256]
        left_bc_cons    = bc_left[:, 0:1, :]   # [B, 1, 1]
        right_bc_cons   = bc_right[:, 0:1, :]  # [B, 1, 1]
        conserved_ext   = self._build_ghost_extended(
            conserved_phys, left_bc_cons, right_bc_cons)  # [B, 1, L+2R]

        # =====================================================================
        # 6. Conservative transport on extended domain (UNCHANGED)
        # =====================================================================
        out_ext, in_ext = self._compute_transport(
            conserved_ext, outflow_pct, outflow_dist,
            inflow_pct, inflow_dist)

        # =====================================================================
        # 7. Combine and crop to physical domain
        # =====================================================================
        combined_ext = (out_ext + in_ext) / 2             # [B, 1, L+2R]

        outflow_change = out_ext[:, :, R:R+L]             # [B, 1, 256]
        inflow_change  = in_ext[:, :, R:R+L]              # [B, 1, 256]
        combined_phys  = combined_ext[:, :, R:R+L]        # [B, 1, 256]

        next_phys = conserved_phys + combined_phys         # [B, 1, 256]

        # =====================================================================
        # 8. Reassemble with BCs at positions 0 and 257
        # =====================================================================
        next_full = torch.cat(
            [left_bc_cons, next_phys, right_bc_cons], dim=-1)  # [B, 1, 258]

        return next_full, outflow_change, inflow_change

