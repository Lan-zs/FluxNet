"""
fluxgnn_traffic_1d.py

FluxGNN-style 1D conservative traffic flow surrogate model.

Target PDE:  ∂ₜρ + ∂ₓ[ρ(1-ρ)] = 0   (LWR traffic flow)

Spatial layout of the 258-length input/output:
    index 0        : left  Dirichlet boundary value  ρ_L
    indices 1..256 : 256 internal finite-volume cells
    index 257      : right Dirichlet boundary value  ρ_R

The model predicts the next time-step field by:
    1.  Encoding cell & boundary features to a hidden representation.
    2.  Running several rounds of face-based message passing.
    3.  Computing a scalar numerical flux at every face (257 faces total)
        via a shared FaceMLP.
    4.  Updating the 256 interior cells with the conservative tally
            ρ_j^{n+1} = ρ_j^n − (F_{j+1/2} − F_{j-1/2})
    5.  Re-attaching the original Dirichlet boundary values.

Two flux computation strategies are supported (selected via `flux_mode`):
    'learned'              – (Scheme A, default) The face MLP directly
                             outputs the numerical flux from enriched
                             face features.
    'physics_correction'   – (Scheme B) A classical Lax-Friedrichs base
                             flux  ½[f(ρ_L)+f(ρ_R)] − ½α(ρ_R−ρ_L)
                             is augmented with a learned correction.

External interface is fully aligned with fno_1d.py / FNO_1D:
    model = FluxGNN_Traffic1D(in_channels=…, out_channels=…, …)
    (next_field,) = model(x)          # x: [B, C_in, 258]
                                       # next_field: [B, C_out, 258]

`prediction_mode` and `bound_mode` are accepted in the constructor for
API compatibility with the existing training pipeline but are **not**
active inside this model (Section VII of the specification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helper: point-wise MLP implemented as stacked 1×1 Conv1d
# ---------------------------------------------------------------------------

def _make_pointwise_mlp(in_ch, out_ch, hidden_ch, n_hidden, act=nn.GELU):
    """Build a small point-wise (per-face / per-cell) MLP using Conv1d(k=1).

    Args:
        in_ch:    input channels
        out_ch:   output channels
        hidden_ch: hidden layer width
        n_hidden:  number of hidden layers  (≥ 1)
        act:       activation class

    Returns:
        nn.Sequential
    """
    layers = [nn.Conv1d(in_ch, hidden_ch, 1), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Conv1d(hidden_ch, hidden_ch, 1), act()]
    layers.append(nn.Conv1d(hidden_ch, out_ch, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# FaceMLP – shared network that maps face features  →  scalar flux(es)
# ---------------------------------------------------------------------------

class FaceMLP(nn.Module):
    """Shared face-flux network.

    For every face the input is the concatenation of the left-state and
    right-state feature vectors (each of dimension ``hidden_dim``).
    The output is one scalar flux per ``out_channels``.

    All operations are point-wise across the face dimension (last axis)
    and are implemented as 1×1 Conv1d for efficiency.
    """

    def __init__(self, hidden_dim: int, out_channels: int,
                 flux_hidden_dim: int = 64, num_hidden_layers: int = 2):
        super().__init__()
        # Input: [B, 2*hidden_dim, n_faces]
        # Output: [B, out_channels, n_faces]
        self.net = _make_pointwise_mlp(
            in_ch=2 * hidden_dim,
            out_ch=out_channels,
            hidden_ch=flux_hidden_dim,
            n_hidden=num_hidden_layers,
        )

    def forward(self, face_features):
        """
        Args:
            face_features: [B, 2*hidden_dim, n_faces]
        Returns:
            flux: [B, out_channels, n_faces]
        """
        return self.net(face_features)


# ---------------------------------------------------------------------------
# MessagePassingLayer – one round of face-based message passing
# ---------------------------------------------------------------------------

class MessagePassingLayer(nn.Module):
    """One round of 1D face-based message passing.

    1. Construct face features by concatenating left/right cell hidden
       states (boundary faces use the encoded BC vector).
    2. Apply a face network to produce per-face messages.
    3. Aggregate messages back to cells:  cell_j receives
       face_msg[j] (left face) + face_msg[j+1] (right face).
    4. Update cell hidden state with a residual connection.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Face network: [B, 2H, 257] → [B, H, 257]
        self.face_net = nn.Sequential(
            nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        # Cell update network (applied after aggregation): [B, H, 256] → [B, H, 256]
        self.cell_update = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )

    def forward(self, h, h_bc_left, h_bc_right):
        """
        Args:
            h:          [B, H, 256]   – current cell hidden states
            h_bc_left:  [B, H, 1]     – encoded left  Dirichlet BC
            h_bc_right: [B, H, 1]     – encoded right Dirichlet BC

        Returns:
            h_updated:  [B, H, 256]   – updated cell hidden states
        """
        # --- build face features (257 faces) ---------------------------------
        #   face j  →  (left_state[j], right_state[j])
        #   face 0:        left = BC_L,       right = cell[0]
        #   face 1..255:   left = cell[j-1],  right = cell[j]
        #   face 256:      left = cell[255],   right = BC_R
        left_states  = torch.cat([h_bc_left,  h], dim=-1)   # [B, H, 257]
        right_states = torch.cat([h, h_bc_right], dim=-1)   # [B, H, 257]
        face_input   = torch.cat([left_states, right_states], dim=1)  # [B, 2H, 257]

        face_msgs = self.face_net(face_input)                # [B, H, 257]

        # --- aggregate to cells -----------------------------------------------
        # cell j receives messages from its left face (j) and right face (j+1)
        cell_agg = face_msgs[:, :, :-1] + face_msgs[:, :, 1:]  # [B, H, 256]

        # --- residual update ---------------------------------------------------
        h_updated = h + self.cell_update(cell_agg)            # [B, H, 256]
        return h_updated


# ---------------------------------------------------------------------------
# FluxGNNTrafficCore – finite-volume update logic
# ---------------------------------------------------------------------------

class FluxGNNTrafficCore(nn.Module):
    """Core finite-volume update module.

    Encodes cell / BC features, runs message passing, computes face
    fluxes, and performs the conservative cell update.

    Args:
        in_channels:       number of input feature channels per cell / BC
        out_channels:      number of conserved output channels (typically 1)
        hidden_dim:        hidden-state width
        num_mp_layers:     number of message-passing rounds
        flux_hidden_dim:   width of the flux MLP hidden layers
        num_flux_hidden:   depth  of the flux MLP hidden layers
        flux_mode:         'learned' (Scheme A) or 'physics_correction' (Scheme B)
    """

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 hidden_dim: int = 64,
                 num_mp_layers: int = 4,
                 flux_hidden_dim: int = 64,
                 num_flux_hidden: int = 2,
                 flux_mode: str = 'learned'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.flux_mode = flux_mode

        # --- encoders ---------------------------------------------------------
        self.cell_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.bc_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )

        # --- message-passing layers -------------------------------------------
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_mp_layers)
        ])

        # --- flux network(s) --------------------------------------------------
        self.flux_mlp = FaceMLP(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            flux_hidden_dim=flux_hidden_dim,
            num_hidden_layers=num_flux_hidden,
        )

        if flux_mode == 'physics_correction':
            # A separate (typically smaller) correction network.  Its output
            # is *added* to the Lax-Friedrichs base flux.
            self.correction_mlp = FaceMLP(
                hidden_dim=hidden_dim,
                out_channels=out_channels,
                flux_hidden_dim=flux_hidden_dim,
                num_hidden_layers=num_flux_hidden,
            )
            # Learnable scaling for the correction so it starts near zero
            self.correction_scale = nn.Parameter(torch.tensor(0.01))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_face_features(h, h_bc_left, h_bc_right):
        """Concatenate left/right hidden states for all 257 faces.

        Returns:
            face_feats: [B, 2*H, 257]
        """
        left_states  = torch.cat([h_bc_left,  h], dim=-1)   # [B, H, 257]
        right_states = torch.cat([h, h_bc_right], dim=-1)   # [B, H, 257]
        return torch.cat([left_states, right_states], dim=1) # [B, 2H, 257]

    @staticmethod
    def _lax_friedrichs_flux(rho_left, rho_right):
        """Classical (global) Lax-Friedrichs numerical flux for f(ρ)=ρ(1-ρ).

        F_LF = 0.5 * [f(ρ_L) + f(ρ_R)]  −  0.5 * α * (ρ_R − ρ_L)

        For f(ρ) = ρ(1-ρ),  max|f'(ρ)| = max|1-2ρ| ≤ 1  over [0,1],
        so we set α = 1.

        Args:
            rho_left:  [B, 1, n_faces]   density at left  side of each face
            rho_right: [B, 1, n_faces]   density at right side of each face

        Returns:
            flux: [B, 1, n_faces]
        """
        f_left  = rho_left  * (1.0 - rho_left)
        f_right = rho_right * (1.0 - rho_right)
        alpha = 1.0  # global maximum wave speed for ρ ∈ [0,1]
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (rho_right - rho_left)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, interior, bc_left, bc_right):
        """
        Args:
            interior: [B, in_channels, 256]   – 256 internal cell features
            bc_left:  [B, in_channels, 1]     – left  Dirichlet BC features
            bc_right: [B, in_channels, 1]     – right Dirichlet BC features

        Returns:
            next_interior: [B, out_channels, 256]  – updated density field
        """
        B = interior.shape[0]

        # ---- encode ----------------------------------------------------------
        h       = self.cell_encoder(interior)   # [B, H, 256]
        h_bc_L  = self.bc_encoder(bc_left)      # [B, H, 1]
        h_bc_R  = self.bc_encoder(bc_right)     # [B, H, 1]

        # ---- message passing -------------------------------------------------
        for mp in self.mp_layers:
            h = mp(h, h_bc_L, h_bc_R)           # [B, H, 256]

        # ---- compute face fluxes (257 faces) ---------------------------------
        face_feats = self._build_face_features(h, h_bc_L, h_bc_R)  # [B, 2H, 257]

        if self.flux_mode == 'learned':
            # Scheme A: pure learned numerical flux
            fluxes = self.flux_mlp(face_feats)                       # [B, C_out, 257]

        elif self.flux_mode == 'physics_correction':
            # Scheme B: Lax-Friedrichs base + learned correction
            # Raw density at each face side (channel 0 = density)
            rho_interior = interior[:, 0:1, :]                        # [B, 1, 256]
            rho_bc_L     = bc_left[:, 0:1, :]                        # [B, 1, 1]
            rho_bc_R     = bc_right[:, 0:1, :]                       # [B, 1, 1]

            rho_left_face  = torch.cat([rho_bc_L, rho_interior], dim=-1)  # [B, 1, 257]
            rho_right_face = torch.cat([rho_interior, rho_bc_R], dim=-1)  # [B, 1, 257]

            base_flux  = self._lax_friedrichs_flux(rho_left_face, rho_right_face)  # [B, 1, 257]
            correction = self.correction_mlp(face_feats)                           # [B, C_out, 257]

            # If out_channels > 1, only the first channel gets the physics base
            if self.out_channels == 1:
                fluxes = base_flux + self.correction_scale * correction
            else:
                fluxes = self.correction_scale * correction
                fluxes[:, 0:1, :] = fluxes[:, 0:1, :] + base_flux

        else:
            raise ValueError(
                f"Unknown flux_mode='{self.flux_mode}'. "
                "Choose 'learned' or 'physics_correction'."
            )

        # ---- conservative cell update ----------------------------------------
        # For cell j:  ρ_j^{n+1} = ρ_j^n − (F_{j+1/2} − F_{j-1/2})
        #   F_{j-1/2} = fluxes[:, :, j]     (left face of cell j)
        #   F_{j+1/2} = fluxes[:, :, j+1]   (right face of cell j)
        net_flux = fluxes[:, :, 1:] - fluxes[:, :, :-1]  # [B, C_out, 256]

        # Only channels 0 : out_channels of the interior participate
        rho_current = interior[:, :self.out_channels, :]  # [B, C_out, 256]
        next_interior = rho_current - net_flux            # [B, C_out, 256]

        return next_interior


# ---------------------------------------------------------------------------
# FluxGNN_Traffic1D – top-level model (interface-compatible with FNO_1D)
# ---------------------------------------------------------------------------

# class FluxGNN_Traffic1D(nn.Module):
class FluxGNN_1D(nn.Module):
    """FluxGNN-style 1D conservative traffic flow surrogate.

    Drop-in replacement for ``FNO_1D`` in the existing training pipeline.

    Constructor Parameters
    ----------------------
    in_channels : int
        Number of input channels (channel 0 = ρ, rest = auxiliary).
    out_channels : int
        Number of output channels (typically 1 for density).
    hidden_dim : int
        Width of the hidden representation in cell / face networks.
    num_mp_layers : int
        Number of face-based message-passing rounds.
    flux_hidden_dim : int
        Hidden width inside the FaceMLP.
    num_flux_hidden : int
        Number of hidden layers inside the FaceMLP.
    flux_mode : str
        ``'learned'`` (Scheme A, default) or ``'physics_correction'`` (Scheme B).
    prediction_mode : str
        Accepted for API compatibility; **not used** in this model.
    bound_mode : str
        Accepted for API compatibility; **not used** in this model.
    lower_bound, upper_bound : float
        Accepted for API compatibility; **not used** in this model.

    Forward I/O
    ------------
    Input:  x  of shape ``[B, in_channels, 258]``
    Output: ``(next_field,)``  where ``next_field.shape == [B, out_channels, 258]``
            Boundary positions 0 and 257 are copied from input unchanged.
    """

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 hidden_dim: int = 32,
                 num_mp_layers: int = 4,
                 flux_hidden_dim: int = 32,
                 num_flux_hidden: int = 2,
                 flux_mode: str = 'learned',
                 # --- API-compat params (not active for FluxGNN) ---
                 prediction_mode: str = 'direct',
                 bound_mode: str = 'none',
                 lower_bound: float = 0.0,
                 upper_bound: float = 1.0,
                 base_channels=64,
                 num_blocks=1,
                 kernel_size=3,
                 modes=16,
                 width=64,
                 num_layers=4,
                 # --- swallow any extra FNO-specific kwargs --------
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.flux_mode = flux_mode

        # Store for API compatibility (unused internally)
        self.prediction_mode = prediction_mode
        self.bound_mode = bound_mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Core finite-volume module
        self.core = FluxGNNTrafficCore(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_mp_layers=num_mp_layers,
            flux_hidden_dim=flux_hidden_dim,
            num_flux_hidden=num_flux_hidden,
            flux_mode=flux_mode,
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, 258]
               x[:, :, 0]       – left  Dirichlet BC  (ρ_L + aux)
               x[:, :, 1:257]   – 256 internal FV cells
               x[:, :, 257]     – right Dirichlet BC  (ρ_R + aux)

        Returns:
            (next_field,) where next_field: [B, out_channels, 258]
               next_field[:, :, 0]     = x[:, :out_channels, 0]     (BC preserved)
               next_field[:, :, 257]   = x[:, :out_channels, 257]   (BC preserved)
               next_field[:, :, 1:257] = conservatively-updated interior
        """
        # --- decompose input --------------------------------------------------
        bc_left   = x[:, :, 0:1]       # [B, C_in, 1]
        interior  = x[:, :, 1:257]     # [B, C_in, 256]
        bc_right  = x[:, :, 257:258]   # [B, C_in, 1]

        # --- core FV update ---------------------------------------------------
        next_interior = self.core(interior, bc_left, bc_right)  # [B, C_out, 256]

        # --- reassemble with original boundaries -----------------------------
        bc_left_out  = x[:, :self.out_channels, 0:1]     # [B, C_out, 1]
        bc_right_out = x[:, :self.out_channels, 257:258]  # [B, C_out, 1]

        next_field = torch.cat(
            [bc_left_out, next_interior, bc_right_out], dim=-1
        )  # [B, C_out, 258]

        return (next_field,)

