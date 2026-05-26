"""
FluxNet Models Package

Model naming convention:
- N: No constraint (fluxes can be positive or negative) - deprecated
- P: Positive constraint (softplus ensures positive fluxes)
- L: Lower bound constraint (values >= lower_bound)
- D: Double bound constraint (lower_bound <= values <= upper_bound)
- U: Upper bound constraint (values <= upper_bound)

Shallow Water head configurations:
- LAP: L-head for h + Advection-Pressure decomposition (RECOMMENDED)
- PAP: P-head for h + Advection-Pressure decomposition
- LAP_no_gate: LAP without h^2 pressure gate
- PPP: P-head for all fields
- LPP: L-head for h, P-head for mx/my
- NNN: No constraint (deprecated)
"""

# 2D FluxNet models
from .fluxnet_n_2d import FluxNet_N
from .fluxnet_p_2d import FluxNet_P
from .fluxnet_l_2d import FluxNet_L
from .fluxnet_d_2d import FluxNet_D

# 1D FluxNet models
from .fluxnet_n_1d import FluxNet_N_1D
from .fluxnet_p_1d import FluxNet_P_1D
from .fluxnet_l_1d import FluxNet_L_1D
from .fluxnet_d_1d import FluxNet_D_1D
from .fluxnet_u_1d import FluxNet_U_1D  # Upper bound only

# Shallow water model
from .fluxnet_sw_lap import FluxNet_SW_2D

# Baseline models
from .cnn_baseline import CNN_Baseline_1D, CNN_Baseline_2D
from .cnn_sw_2d_baseline import FluxNet_SW_Baseline

# FNO models
try:
    from .fno_sw import FNO_SW
except ImportError:
    FNO_SW = None

try:
    from .fno_1d import FNO_1D, FNO_FluxD_1D
except ImportError:
    FNO_1D = None
    FNO_FluxD_1D = None

# Shallow water baselines with projection
try:
    from .sw_baselines import FNO_SW_Proj, CNN_SW_Proj
except ImportError:
    FNO_SW_Proj = None
    CNN_SW_Proj = None

# FNO with FluxLAP head (for ablation study)
try:
    from .fno_sw_fluxlap import FNO_FluxLAP
except ImportError:
    FNO_FluxLAP = None

# Dirichlet boundary models (for traffic flow)
try:
    from .fluxnet_d_1d_boundary import FluxNet_D_Dirichlet_1D
except ImportError:
    FluxNet_D_Dirichlet_1D = None

try:
    from .fluxgnn_1d_boundary import FluxGNN_1D
except ImportError:
    FluxGNN_1D = None

try:
    from .fluxgnn_d_1d_boundary import FluxGNN_D_1D
except ImportError:
    FluxGNN_D_1D = None

# Backward compatibility aliases (deprecated, use new names)
FluxNet_U = FluxNet_N
FluxNet_U_unbounded = FluxNet_N  # Clarify that old U means unbounded, not upper-bounded

__all__ = [
    # 2D FluxNet
    'FluxNet_N',
    'FluxNet_P',
    'FluxNet_L',
    'FluxNet_D',

    # 1D FluxNet
    'FluxNet_N_1D',
    'FluxNet_P_1D',
    'FluxNet_L_1D',
    'FluxNet_D_1D',
    'FluxNet_U_1D',  # Upper bound only

    # Shallow water
    'FluxNet_SW_2D',
    'FluxNet_SW_Baseline',

    # FNO
    'FNO_SW',
    'FNO_1D',
    'FNO_FluxD_1D',

    # Shallow water baselines with projection
    'FNO_SW_Proj',
    'CNN_SW_Proj',

    # FNO with FluxLAP head (for ablation)
    'FNO_FluxLAP',

    # Dirichlet boundary / FluxGNN models
    'FluxNet_D_Dirichlet_1D',
    'FluxGNN_1D',
    'FluxGNN_D_1D',

    # Baselines
    'CNN_Baseline_1D',
    'CNN_Baseline_2D',

    # Backward compatibility (deprecated)
    'FluxNet_U',
    'FluxNet_U_unbounded',
]
