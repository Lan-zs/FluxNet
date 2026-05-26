"""
2D Shallow Water Equations Dataset Generator
=============================================
Model:
    ∂h/∂t + ∂(mx)/∂x + ∂(my)/∂y = 0
    ∂mx/∂t + ∂(mx²/h + gh²/2)/∂x + ∂(mx·my/h)/∂y = 0
    ∂my/∂t + ∂(mx·my/h)/∂x + ∂(my²/h + gh²/2)/∂y = 0

Conserved variables: (h, mx, my) - water depth, x-momentum, y-momentum

Features:
- Double periodic boundary conditions
- Flat bottom (b=0), no friction, no Coriolis
- Fixed time step for ALL simulations
- Challenging initial conditions with dry regions (h=0)
- Stratified sampling across 4 case types

Initial Conditions:
- CaseA1: Multiple Gaussian bumps/depressions (zero momentum)
- CaseA2: Random low-frequency sinusoidal perturbations (zero momentum)
- CaseB1: Multiple Gaussian bumps/depressions with initial velocity field
- CaseB2: Random low-frequency sinusoidal perturbations with initial velocity field

Numerical method: Finite Volume + Rusanov flux + SSP-RK3
=============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Global Parameters
# ============================================================
g = 9.81  # Gravity acceleration
H0 = 1.0  # Mean water depth reference

# Grid parameters (high resolution for computation)
Nx, Ny = 128, 128

Lx, Ly = 10.0, 10.0
dx, dy = Lx / Nx, Ly / Ny

# Simulation parameters
T_final = 2.4  # Final simulation time (same for all samples)
FIXED_DT = 0.004  # Fixed time step for ALL simulations and ALL time steps

# Downsampling parameters for saving
TIME_DOWNSAMPLE = 10  # Save every TIME_DOWNSAMPLE steps
SPACE_DOWNSAMPLE = 2  # Spatial coarsening factor (128/2 = 64 saved)

# Grid coordinates (high resolution)
x = np.linspace(dx / 2, Lx - dx / 2, Nx)
y = np.linspace(dy / 2, Ly - dy / 2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Output directories
BASE_DIR = Path("shallow_water_dataset")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
FIG_DIR = BASE_DIR / "figures"

# Global random seed
GLOBAL_SEED = 42

# Dataset split configuration
# Train: 50, Val: 20, Test: 50
# Ratio A1:A2:B1:B2 = 10:10:15:15


CATEGORY_CONFIG = {
    'CaseA1': {'total': 24, 'train': 10, 'val': 4, 'test': 10},
    'CaseA2': {'total': 24, 'train': 10, 'val': 4, 'test': 10},
    'CaseB1': {'total': 36, 'train': 15, 'val': 6, 'test': 15},
    'CaseB2': {'total': 36, 'train': 15, 'val': 6, 'test': 15},
}

# CATEGORY_CONFIG = {
#     'CaseA1': {'total': 10, 'train': 0, 'val': 0, 'test': 10},
#     'CaseA2': {'total': 10, 'train': 0, 'val': 0, 'test': 10},
#     'CaseB1': {'total': 15, 'train': 0, 'val': 0, 'test': 15},
#     'CaseB2': {'total': 15, 'train': 0, 'val': 0, 'test': 15},
# }



# ============================================================
# Numerical Flux Functions (Vectorized)
# ============================================================

def rusanov_flux_x(hL, mxL, myL, hR, mxR, myR):
    """Rusanov (Local Lax-Friedrichs) numerical flux in x-direction"""
    eps = 1e-10

    # Left state flux
    hL_s = np.maximum(hL, eps)
    uL = mxL / hL_s
    F1L = mxL
    F2L = mxL * uL + 0.5 * g * hL ** 2
    F3L = myL * uL

    # Right state flux
    hR_s = np.maximum(hR, eps)
    uR = mxR / hR_s
    F1R = mxR
    F2R = mxR * uR + 0.5 * g * hR ** 2
    F3R = myR * uR

    # Maximum local wave speed
    cL = np.sqrt(g * hL_s)
    cR = np.sqrt(g * hR_s)
    alpha = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    # Rusanov flux
    F1 = 0.5 * (F1L + F1R) - 0.5 * alpha * (hR - hL)
    F2 = 0.5 * (F2L + F2R) - 0.5 * alpha * (mxR - mxL)
    F3 = 0.5 * (F3L + F3R) - 0.5 * alpha * (myR - myL)

    return F1, F2, F3


def rusanov_flux_y(hL, mxL, myL, hR, mxR, myR):
    """Rusanov numerical flux in y-direction"""
    eps = 1e-10

    # Bottom state flux
    hL_s = np.maximum(hL, eps)
    vL = myL / hL_s
    G1L = myL
    G2L = mxL * vL
    G3L = myL * vL + 0.5 * g * hL ** 2

    # Top state flux
    hR_s = np.maximum(hR, eps)
    vR = myR / hR_s
    G1R = myR
    G2R = mxR * vR
    G3R = myR * vR + 0.5 * g * hR ** 2

    # Maximum local wave speed
    cL = np.sqrt(g * hL_s)
    cR = np.sqrt(g * hR_s)
    alpha = np.maximum(np.abs(vL) + cL, np.abs(vR) + cR)

    # Rusanov flux
    G1 = 0.5 * (G1L + G1R) - 0.5 * alpha * (hR - hL)
    G2 = 0.5 * (G2L + G2R) - 0.5 * alpha * (mxR - mxL)
    G3 = 0.5 * (G3L + G3R) - 0.5 * alpha * (myR - myL)

    return G1, G2, G3


def compute_rhs(h, mx, my):
    """Compute right-hand side: -∂F/∂x - ∂G/∂y (vectorized)"""
    # x-direction interface (i+1/2, j)
    hL_x = h
    hR_x = np.roll(h, -1, axis=0)
    mxL_x = mx
    mxR_x = np.roll(mx, -1, axis=0)
    myL_x = my
    myR_x = np.roll(my, -1, axis=0)

    Fx1, Fx2, Fx3 = rusanov_flux_x(hL_x, mxL_x, myL_x, hR_x, mxR_x, myR_x)

    # y-direction interface (i, j+1/2)
    hL_y = h
    hR_y = np.roll(h, -1, axis=1)
    mxL_y = mx
    mxR_y = np.roll(mx, -1, axis=1)
    myL_y = my
    myR_y = np.roll(my, -1, axis=1)

    Gy1, Gy2, Gy3 = rusanov_flux_y(hL_y, mxL_y, myL_y, hR_y, mxR_y, myR_y)

    # Compute divergence (periodic boundary handled by roll)
    dFx1 = (Fx1 - np.roll(Fx1, 1, axis=0)) / dx
    dFx2 = (Fx2 - np.roll(Fx2, 1, axis=0)) / dx
    dFx3 = (Fx3 - np.roll(Fx3, 1, axis=0)) / dx

    dGy1 = (Gy1 - np.roll(Gy1, 1, axis=1)) / dy
    dGy2 = (Gy2 - np.roll(Gy2, 1, axis=1)) / dy
    dGy3 = (Gy3 - np.roll(Gy3, 1, axis=1)) / dy

    rhs_h = -(dFx1 + dGy1)
    rhs_mx = -(dFx2 + dGy2)
    rhs_my = -(dFx3 + dGy3)

    return rhs_h, rhs_mx, rhs_my


def ssprk3_step(h, mx, my, dt):
    """SSP-RK3 (Strong Stability Preserving Runge-Kutta 3rd order) time stepping"""
    # Stage 1
    rhs_h1, rhs_mx1, rhs_my1 = compute_rhs(h, mx, my)
    h1 = h + dt * rhs_h1
    mx1 = mx + dt * rhs_mx1
    my1 = my + dt * rhs_my1

    # Stage 2
    rhs_h2, rhs_mx2, rhs_my2 = compute_rhs(h1, mx1, my1)
    h2 = 0.75 * h + 0.25 * (h1 + dt * rhs_h2)
    mx2 = 0.75 * mx + 0.25 * (mx1 + dt * rhs_mx2)
    my2 = 0.75 * my + 0.25 * (my1 + dt * rhs_my2)

    # Stage 3
    rhs_h3, rhs_mx3, rhs_my3 = compute_rhs(h2, mx2, my2)
    h_new = (1.0 / 3.0) * h + (2.0 / 3.0) * (h2 + dt * rhs_h3)
    mx_new = (1.0 / 3.0) * mx + (2.0 / 3.0) * (mx2 + dt * rhs_mx3)
    my_new = (1.0 / 3.0) * my + (2.0 / 3.0) * (my2 + dt * rhs_my3)

    return h_new, mx_new, my_new


# ============================================================
# Main Simulation Function (Fixed Time Step)
# ============================================================

def simulate_swe(h0, mx0, my0, T_final, dt, time_downsample=1):
    """
    Main simulation loop with FIXED time step

    Parameters:
        h0, mx0, my0: initial conditions (Nx, Ny)
        T_final: final simulation time
        dt: FIXED time step (same for all steps)
        time_downsample: save every time_downsample steps

    Returns:
        times: saved time array
        h_history, mx_history, my_history: saved field histories
        mass_history, momx_history, momy_history: conserved quantity histories
        h_min_history, h_max_history: min/max water depth histories
    """
    h = h0.copy()
    mx = mx0.copy()
    my = my0.copy()

    n_steps = int(np.ceil(T_final / dt))

    # Storage for saved states
    times_saved = []
    h_history = []
    mx_history = []
    my_history = []

    # Conservation tracking (at ALL steps)
    mass_history = []
    momx_history = []
    momy_history = []
    h_min_history = []
    h_max_history = []

    cell_area = dx * dy

    def record_conservation(t, h, mx, my):
        mass = np.sum(h) * cell_area
        momx = np.sum(mx) * cell_area
        momy = np.sum(my) * cell_area
        mass_history.append(mass)
        momx_history.append(momx)
        momy_history.append(momy)
        h_min_history.append(np.min(h))
        h_max_history.append(np.max(h))

    # Save initial state
    times_saved.append(0.0)
    h_history.append(h.copy())
    mx_history.append(mx.copy())
    my_history.append(my.copy())
    record_conservation(0.0, h, mx, my)

    t = 0.0
    save_counter = 0

    for step in range(n_steps):
        # Adjust last step if needed
        current_dt = min(dt, T_final - t)
        if current_dt <= 0:
            break

        # Time stepping (NO clipping of h during evolution)
        h, mx, my = ssprk3_step(h, mx, my, current_dt)
        t += current_dt
        save_counter += 1

        # Record conservation at every step
        record_conservation(t, h, mx, my)

        # Save state at downsampled intervals
        if save_counter >= time_downsample:
            times_saved.append(t)
            h_history.append(h.copy())
            mx_history.append(mx.copy())
            my_history.append(my.copy())
            save_counter = 0

    # Ensure final state is saved
    if len(times_saved) == 0 or times_saved[-1] < t - 1e-10:
        times_saved.append(t)
        h_history.append(h.copy())
        mx_history.append(mx.copy())
        my_history.append(my.copy())

    return (np.array(times_saved),
            np.array(h_history),
            np.array(mx_history),
            np.array(my_history),
            np.array(mass_history),
            np.array(momx_history),
            np.array(momy_history),
            np.array(h_min_history),
            np.array(h_max_history))


# ============================================================
# Spatial Downsampling (Box Averaging - Conserves Mass & Momentum)
# ============================================================

def downsample_spatial_conservative(field_history, factor):
    """
    Downsample spatial resolution using box averaging (coarse graining)

    This method preserves total integral:
        sum(field_coarse * dx_coarse * dy_coarse) = sum(field_fine * dx_fine * dy_fine)

    For conserved quantities (h, mx, my), box averaging is the correct method
    because it computes the average density in each coarse cell, and when
    multiplied by the larger cell area, gives the same total.

    Parameters:
        field_history: (n_times, Nx_fine, Ny_fine)
        factor: downsampling factor

    Returns:
        field_coarse: (n_times, Nx_coarse, Ny_coarse)
    """
    n_times, Nx_fine, Ny_fine = field_history.shape
    Nx_coarse = Nx_fine // factor
    Ny_coarse = Ny_fine // factor

    # Reshape and average
    field_reshaped = field_history.reshape(
        n_times, Nx_coarse, factor, Ny_coarse, factor
    )
    field_coarse = np.mean(field_reshaped, axis=(2, 4))

    return field_coarse


def get_coarse_grid(factor):
    """Get downsampled grid coordinates"""
    Nx_coarse = Nx // factor
    Ny_coarse = Ny // factor
    dx_coarse = Lx / Nx_coarse
    dy_coarse = Ly / Ny_coarse

    x_coarse = np.linspace(dx_coarse / 2, Lx - dx_coarse / 2, Nx_coarse)
    y_coarse = np.linspace(dy_coarse / 2, Ly - dy_coarse / 2, Ny_coarse)

    return x_coarse, y_coarse, dx_coarse, dy_coarse


# ============================================================
# Helper Functions for Initial Conditions
# ============================================================

def periodic_distance(X, Y, x0, y0, Lx, Ly):
    """Compute periodic distance from (x0, y0)"""
    dx_wrap = X - x0
    dy_wrap = Y - y0

    dx_wrap = np.where(dx_wrap > Lx / 2, dx_wrap - Lx, dx_wrap)
    dx_wrap = np.where(dx_wrap < -Lx / 2, dx_wrap + Lx, dx_wrap)
    dy_wrap = np.where(dy_wrap > Ly / 2, dy_wrap - Ly, dy_wrap)
    dy_wrap = np.where(dy_wrap < -Ly / 2, dy_wrap + Ly, dy_wrap)

    return dx_wrap, dy_wrap


def gaussian_bump(X, Y, x0, y0, sigma, amplitude, Lx=Lx, Ly=Ly):
    """Gaussian bump/depression with periodic wrapping"""
    dx_wrap, dy_wrap = periodic_distance(X, Y, x0, y0, Lx, Ly)
    r2 = dx_wrap ** 2 + dy_wrap ** 2
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def smooth_random_field_2d(rng, Nx, Ny, n_modes, k_max, Lx, Ly):
    """
    Generate smooth random field using low-frequency Fourier synthesis

    Parameters:
        rng: numpy random generator
        Nx, Ny: grid dimensions
        n_modes: number of random Fourier modes
        k_max: maximum wavenumber index
        Lx, Ly: domain size

    Returns:
        field: (Nx, Ny) smooth random field with zero mean
    """
    field = np.zeros((Nx, Ny))

    for _ in range(n_modes):
        # Random wavenumber (low frequency)
        kx_idx = rng.integers(-k_max, k_max + 1)
        ky_idx = rng.integers(-k_max, k_max + 1)

        if kx_idx == 0 and ky_idx == 0:
            continue

        kx = 2 * np.pi * kx_idx / Lx
        ky = 2 * np.pi * ky_idx / Ly

        # Random amplitude and phase
        amp = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, 2 * np.pi)

        # Add mode
        field += amp * np.sin(kx * X + ky * Y + phase)

    # Normalize to unit std
    if np.std(field) > 1e-10:
        field = field / np.std(field)

    return field


def generate_complex_velocity_field(rng, Nx, Ny, Lx, Ly, n_modes=15, k_max=4):
    """
    Generate complex but smooth velocity field using random low-frequency Fourier synthesis

    Returns u, v fields with:
    - Smooth spatial variation
    - Mixed positive/negative values
    - Non-trivial spatial structure
    """
    # Generate independent random fields for u and v
    u_raw = smooth_random_field_2d(rng, Nx, Ny, n_modes, k_max, Lx, Ly)
    v_raw = smooth_random_field_2d(rng, Nx, Ny, n_modes + 3, k_max, Lx, Ly)

    # Scale to reasonable velocity magnitude
    # Typical gravity wave speed ~ sqrt(g*H0) ~ 3.1 m/s
    vel_scale = rng.uniform(0.1, 0.4) * np.sqrt(g * H0)

    u = vel_scale * u_raw
    v = vel_scale * v_raw

    return u, v


# ============================================================
# Initial Condition Generators
# ============================================================

def generate_case_A1(rng):
    """
    Case A1: Multiple Gaussian bumps/depressions (zero momentum)

    Features:
    - 4-12 random Gaussians with mixed positive/negative amplitudes
    - Global negative bias to encourage dry regions
    - Wide-scale negative Gaussians for connected dry areas
    """
    n_bumps = rng.integers(4, 13)  # 4 to 12 bumps

    # Target dry fraction
    target_dry_frac = rng.uniform(0.05, 0.35)

    # Base level (biased negative for dry regions)
    bias = rng.uniform(0.4, 0.9)

    h_raw = np.ones((Nx, Ny)) * bias

    # Add multiple Gaussians
    for _ in range(n_bumps):
        x0 = rng.uniform(0.5, Lx - 0.5)
        y0 = rng.uniform(0.5, Ly - 0.5)
        sigma = rng.uniform(0.3, 1.0)
        amp = rng.uniform(-0.5, 0.5)  # Mixed positive/negative
        h_raw += gaussian_bump(X, Y, x0, y0, sigma, amp)

    # Add 1-3 wide negative Gaussians for connected dry regions
    n_wide = rng.integers(1, 4)
    for _ in range(n_wide):
        x0 = rng.uniform(1, Lx - 1)
        y0 = rng.uniform(1, Ly - 1)
        sigma_wide = rng.uniform(1.5, 3.0)
        amp_neg = rng.uniform(-0.6, -0.2)
        h_raw += gaussian_bump(X, Y, x0, y0, sigma_wide, amp_neg)

    # Adjust bias iteratively to approach target dry fraction
    for _ in range(5):
        h_test = np.clip(h_raw, 0, None)
        current_dry = np.mean(h_test == 0)
        if current_dry < target_dry_frac - 0.05:
            h_raw -= 0.1
        elif current_dry > target_dry_frac + 0.1:
            h_raw += 0.1
        else:
            break

    # Final clip
    h = np.clip(h_raw, 0, None)

    mx = np.zeros_like(h)
    my = np.zeros_like(h)

    dry_frac = np.mean(h == 0)

    params = {
        'n_bumps': n_bumps, 'n_wide_depressions': n_wide,
        'bias': bias, 'target_dry_frac': target_dry_frac, 'dry_frac': dry_frac
    }

    return h, mx, my, params


def generate_case_A2(rng):
    """
    Case A2: Random low-frequency sinusoidal perturbations (zero momentum)

    Features:
    - Multiple low-frequency Fourier modes
    - Global bias + Gaussian depressions for dry regions
    """
    n_modes = rng.integers(5, 15)

    # Target dry fraction
    target_dry_frac = rng.uniform(0.05, 0.30)

    # Base level
    bias = rng.uniform(0.5, 1.0)

    h_raw = np.ones((Nx, Ny)) * bias

    # Add low-frequency modes
    epsilon = rng.uniform(0.2, 0.5)
    for _ in range(n_modes):
        kx_idx = rng.integers(1, 6)
        ky_idx = rng.integers(1, 6)
        kx = 2 * np.pi * kx_idx / Lx
        ky = 2 * np.pi * ky_idx / Ly
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(-1, 1) * epsilon / np.sqrt(n_modes)
        h_raw += amp * np.sin(kx * X + ky * Y + phase)

    # Add 1-2 wide Gaussian depressions for connected dry areas
    n_depressions = rng.integers(1, 3)
    for _ in range(n_depressions):
        x0 = rng.uniform(1, Lx - 1)
        y0 = rng.uniform(1, Ly - 1)
        sigma = rng.uniform(1.2, 2.5)
        amp = rng.uniform(-0.5, -0.2)
        h_raw += gaussian_bump(X, Y, x0, y0, sigma, amp)

    # Adjust to approach target dry fraction
    for _ in range(5):
        h_test = np.clip(h_raw, 0, None)
        current_dry = np.mean(h_test == 0)
        if current_dry < target_dry_frac - 0.05:
            h_raw -= 0.08
        elif current_dry > target_dry_frac + 0.1:
            h_raw += 0.08
        else:
            break

    h = np.clip(h_raw, 0, None)

    mx = np.zeros_like(h)
    my = np.zeros_like(h)

    dry_frac = np.mean(h == 0)

    params = {
        'n_modes': n_modes, 'epsilon': epsilon, 'n_depressions': n_depressions,
        'bias': bias, 'target_dry_frac': target_dry_frac, 'dry_frac': dry_frac
    }

    return h, mx, my, params


def generate_case_B1(rng):
    """
    Case B1: Multiple Gaussian bumps/depressions with initial velocity field

    Features:
    - Same h field as CaseA1 (multiple Gaussians)
    - Complex smooth velocity field (low-frequency Fourier synthesis)
    - Dry regions with connected areas
    - mx = h*u, my = h*v
    """
    n_bumps = rng.integers(4, 13)  # 4 to 12 bumps

    # Target dry fraction
    target_dry_frac = rng.uniform(0.05, 0.35)

    # Base level (biased negative for dry regions)
    bias = rng.uniform(0.4, 0.9)

    h_raw = np.ones((Nx, Ny)) * bias

    # Add multiple Gaussians
    for _ in range(n_bumps):
        x0 = rng.uniform(0.5, Lx - 0.5)
        y0 = rng.uniform(0.5, Ly - 0.5)
        sigma = rng.uniform(0.3, 1.0)
        amp = rng.uniform(-0.5, 0.5)  # Mixed positive/negative
        h_raw += gaussian_bump(X, Y, x0, y0, sigma, amp)

    # Add 1-3 wide negative Gaussians for connected dry regions
    n_wide = rng.integers(1, 4)
    for _ in range(n_wide):
        x0 = rng.uniform(1, Lx - 1)
        y0 = rng.uniform(1, Ly - 1)
        sigma_wide = rng.uniform(1.5, 3.0)
        amp_neg = rng.uniform(-0.6, -0.2)
        h_raw += gaussian_bump(X, Y, x0, y0, sigma_wide, amp_neg)

    # Adjust bias iteratively to approach target dry fraction
    for _ in range(5):
        h_test = np.clip(h_raw, 0, None)
        current_dry = np.mean(h_test == 0)
        if current_dry < target_dry_frac - 0.05:
            h_raw -= 0.1
        elif current_dry > target_dry_frac + 0.1:
            h_raw += 0.1
        else:
            break

    # Final clip
    h = np.clip(h_raw, 0, None)

    # Generate complex velocity field using random low-frequency Fourier synthesis
    n_vel_modes = rng.integers(10, 20)
    k_max = rng.integers(3, 6)
    u, v = generate_complex_velocity_field(rng, Nx, Ny, Lx, Ly, n_vel_modes, k_max)

    # Compute momentum (mx = h*u, my = h*v)
    # In dry regions (h=0), momentum is also 0
    mx = h * u
    my = h * v

    dry_frac = np.mean(h == 0)

    params = {
        'n_bumps': n_bumps, 'n_wide_depressions': n_wide,
        'n_vel_modes': n_vel_modes, 'k_max_vel': k_max,
        'bias': bias, 'target_dry_frac': target_dry_frac, 'dry_frac': dry_frac,
        'max_u': float(np.max(np.abs(u))), 'max_v': float(np.max(np.abs(v)))
    }

    return h, mx, my, params


def generate_case_B2(rng):
    """
    Case B2: Random low-frequency sinusoidal perturbations with initial velocity field

    Features:
    - Same h field as CaseA2 (sinusoidal modes)
    - Complex smooth velocity field (low-frequency Fourier synthesis)
    - Dry regions with connected areas
    - mx = h*u, my = h*v
    """
    n_modes = rng.integers(5, 15)

    # Target dry fraction
    target_dry_frac = rng.uniform(0.05, 0.30)

    # Base level
    bias = rng.uniform(0.5, 1.0)

    h_raw = np.ones((Nx, Ny)) * bias

    # Add low-frequency modes
    epsilon = rng.uniform(0.2, 0.5)
    for _ in range(n_modes):
        kx_idx = rng.integers(1, 6)
        ky_idx = rng.integers(1, 6)
        kx = 2 * np.pi * kx_idx / Lx
        ky = 2 * np.pi * ky_idx / Ly
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(-1, 1) * epsilon / np.sqrt(n_modes)
        h_raw += amp * np.sin(kx * X + ky * Y + phase)

    # Add 1-2 wide Gaussian depressions for connected dry areas
    n_depressions = rng.integers(1, 3)
    for _ in range(n_depressions):
        x0 = rng.uniform(1, Lx - 1)
        y0 = rng.uniform(1, Ly - 1)
        sigma = rng.uniform(1.2, 2.5)
        amp = rng.uniform(-0.5, -0.2)
        h_raw += gaussian_bump(X, Y, x0, y0, sigma, amp)

    # Adjust to approach target dry fraction
    for _ in range(5):
        h_test = np.clip(h_raw, 0, None)
        current_dry = np.mean(h_test == 0)
        if current_dry < target_dry_frac - 0.05:
            h_raw -= 0.08
        elif current_dry > target_dry_frac + 0.1:
            h_raw += 0.08
        else:
            break

    h = np.clip(h_raw, 0, None)

    # Generate complex velocity field using random low-frequency Fourier synthesis
    n_vel_modes = rng.integers(10, 20)
    k_max = rng.integers(3, 6)
    u, v = generate_complex_velocity_field(rng, Nx, Ny, Lx, Ly, n_vel_modes, k_max)

    # Compute momentum (mx = h*u, my = h*v)
    # In dry regions (h=0), momentum is also 0
    mx = h * u
    my = h * v

    dry_frac = np.mean(h == 0)

    params = {
        'n_modes': n_modes, 'epsilon': epsilon, 'n_depressions': n_depressions,
        'n_vel_modes': n_vel_modes, 'k_max_vel': k_max,
        'bias': bias, 'target_dry_frac': target_dry_frac, 'dry_frac': dry_frac,
        'max_u': float(np.max(np.abs(u))), 'max_v': float(np.max(np.abs(v)))
    }

    return h, mx, my, params


# Generator mapping
GENERATORS = {
    'CaseA1': generate_case_A1,
    'CaseA2': generate_case_A2,
    'CaseB1': generate_case_B1,
    'CaseB2': generate_case_B2,
}


# ============================================================
# Data Saving Functions
# ============================================================

def save_sample_h5(filepath, h_data, mx_data, my_data, x_grid, y_grid, times,
                   category, sample_idx, params,
                   mass_history, momx_history, momy_history,
                   h_min_history, h_max_history, metadata):
    """
    Save a single sample to HDF5 file

    Structure:
        - h: water depth evolution (Nt_saved, Nx_saved, Ny_saved)
        - mx: x-momentum evolution (Nt_saved, Nx_saved, Ny_saved)
        - my: y-momentum evolution (Nt_saved, Nx_saved, Ny_saved)
        - x, y: spatial grids
        - t: time array
        - mass, momx, momy: conservation histories
        - metadata: simulation parameters
    """
    with h5py.File(filepath, 'w') as f:
        # Primary data (float32 for storage efficiency)
        f.create_dataset('h', data=h_data.astype(np.float32), compression='gzip')
        f.create_dataset('mx', data=mx_data.astype(np.float32), compression='gzip')
        f.create_dataset('my', data=my_data.astype(np.float32), compression='gzip')

        # Grids
        f.create_dataset('x', data=x_grid.astype(np.float32))
        f.create_dataset('y', data=y_grid.astype(np.float32))
        f.create_dataset('t', data=times.astype(np.float32))

        # Conservation quantities
        f.create_dataset('mass', data=mass_history.astype(np.float32))
        f.create_dataset('momx', data=momx_history.astype(np.float32))
        f.create_dataset('momy', data=momy_history.astype(np.float32))
        f.create_dataset('h_min', data=h_min_history.astype(np.float32))
        f.create_dataset('h_max', data=h_max_history.astype(np.float32))

        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['Lx'] = Lx
        meta.attrs['Ly'] = Ly
        meta.attrs['Nx_original'] = Nx
        meta.attrs['Ny_original'] = Ny
        meta.attrs['Nx_saved'] = len(x_grid)
        meta.attrs['Ny_saved'] = len(y_grid)
        meta.attrs['dx_saved'] = x_grid[1] - x_grid[0] if len(x_grid) > 1 else Lx
        meta.attrs['dy_saved'] = y_grid[1] - y_grid[0] if len(y_grid) > 1 else Ly
        meta.attrs['T_final'] = T_final
        meta.attrs['dt'] = FIXED_DT
        meta.attrs['Nt_saved'] = len(times)
        meta.attrs['g'] = g
        meta.attrs['category'] = category
        meta.attrs['sample_idx'] = sample_idx
        meta.attrs['time_downsample'] = TIME_DOWNSAMPLE
        meta.attrs['space_downsample'] = SPACE_DOWNSAMPLE

        # Case-specific parameters
        params_grp = f.create_group('params')
        for key, val in params.items():
            if isinstance(val, (int, float, str, bool)):
                params_grp.attrs[key] = val

        # Additional metadata
        for key, val in metadata.items():
            if isinstance(val, (int, float, str, bool)):
                meta.attrs[key] = val


# ============================================================
# Visualization Functions
# ============================================================

def visualize_sample(h5_filepath, save_path):
    """
    Generate a combined 3-panel visualization figure:
    1. 5 time snapshots of h, mx, my (5 columns x 3 rows)
    2. h_min and h_max vs time
    3. Mass and momentum conservation (normalized)

    All text in English, proper symbols
    """
    # Read data
    with h5py.File(h5_filepath, 'r') as f:
        h_data = f['h'][:]
        mx_data = f['mx'][:]
        my_data = f['my'][:]
        x_grid = f['x'][:]
        y_grid = f['y'][:]
        times = f['t'][:]
        mass = f['mass'][:]
        momx = f['momx'][:]
        momy = f['momy'][:]
        h_min = f['h_min'][:]
        h_max = f['h_max'][:]
        category = f['metadata'].attrs['category']
        sample_idx = f['metadata'].attrs['sample_idx']
        dry_frac = f['params'].attrs.get('dry_frac', 0.0)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 16))

    # ==================== Panel 1: 5-time snapshots (3 rows x 5 cols) ====================
    # Select 5 time indices (including first and last)
    n_times = len(times)
    if n_times >= 5:
        time_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]
    else:
        time_indices = list(range(n_times))

    n_cols = len(time_indices)

    # Create gridspec for the snapshot panel
    gs_top = fig.add_gridspec(3, n_cols, left=0.05, right=0.95, top=0.95, bottom=0.45,
                              hspace=0.25, wspace=0.15)

    # Find global ranges for consistent coloring
    h_vmin, h_vmax = 0, np.max(h_data)
    mx_abs_max = np.max(np.abs(mx_data)) + 1e-10
    my_abs_max = np.max(np.abs(my_data)) + 1e-10

    X_plot, Y_plot = np.meshgrid(x_grid, y_grid, indexing='ij')

    for col, tidx in enumerate(time_indices):
        t_val = times[tidx]

        # h (row 0)
        ax = fig.add_subplot(gs_top[0, col])
        im = ax.pcolormesh(X_plot, Y_plot, h_data[tidx], cmap='viridis',
                           vmin=h_vmin, vmax=h_vmax, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(f't = {t_val:.2f}', fontsize=10)
        if col == 0:
            ax.set_ylabel('$h$ (depth)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        # mx (row 1)
        ax = fig.add_subplot(gs_top[1, col])
        im = ax.pcolormesh(X_plot, Y_plot, mx_data[tidx], cmap='RdBu_r',
                           vmin=-mx_abs_max, vmax=mx_abs_max, shading='auto')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('$m_x$ (x-momentum)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        # my (row 2)
        ax = fig.add_subplot(gs_top[2, col])
        im = ax.pcolormesh(X_plot, Y_plot, my_data[tidx], cmap='RdBu_r',
                           vmin=-my_abs_max, vmax=my_abs_max, shading='auto')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('$m_y$ (y-momentum)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    # ==================== Panel 2: h_min/h_max vs time ====================
    ax2 = fig.add_axes([0.08, 0.22, 0.38, 0.18])

    # Time array for conservation (full resolution)
    t_conservation = np.linspace(0, T_final, len(h_min))

    ax2.plot(t_conservation, h_max, 'b-', linewidth=1.5, label='$h_{max}$')
    ax2.plot(t_conservation, h_min, 'r-', linewidth=1.5, label='$h_{min}$')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='$h=0$ bound')
    ax2.fill_between(t_conservation, h_min, 0, where=(h_min < 0),
                     color='red', alpha=0.3, label='Violation region')

    ax2.set_xlabel('Time $t$', fontsize=11)
    ax2.set_ylabel('Water depth $h$', fontsize=11)
    ax2.set_title(f'Water Depth Range vs Time (min(h) = {np.min(h_min):.6f})', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, T_final])

    # ==================== Panel 3: Conservation check ====================
    ax3 = fig.add_axes([0.55, 0.22, 0.38, 0.18])

    # Normalize to initial values
    M0 = mass[0] if mass[0] != 0 else 1.0
    Px0 = momx[0] if np.abs(momx[0]) > 1e-12 else 1.0
    Py0 = momy[0] if np.abs(momy[0]) > 1e-12 else 1.0

    mass_norm = mass / M0

    # For momentum, use absolute change if initial is ~0
    if np.abs(momx[0]) > 1e-12:
        momx_norm = momx / Px0
        momx_label = '$M_x(t)/M_x(0)$'
    else:
        momx_norm = momx - momx[0]
        momx_label = '$M_x(t) - M_x(0)$'

    if np.abs(momy[0]) > 1e-12:
        momy_norm = momy / Py0
        momy_label = '$M_y(t)/M_y(0)$'
    else:
        momy_norm = momy - momy[0]
        momy_label = '$M_y(t) - M_y(0)$'

    ax3.plot(t_conservation, mass_norm, 'b-', linewidth=1.5, label='Mass $M(t)/M(0)$')

    # Plot momentum on twin axis if normalized differently
    if np.abs(momx[0]) > 1e-12 or np.abs(momy[0]) > 1e-12:
        ax3.plot(t_conservation, momx_norm, 'r--', linewidth=1.5, label=momx_label)
        ax3.plot(t_conservation, momy_norm, 'g-.', linewidth=1.5, label=momy_label)
    else:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(t_conservation, momx_norm, 'r--', linewidth=1.5, label=momx_label)
        ax3_twin.plot(t_conservation, momy_norm, 'g-.', linewidth=1.5, label=momy_label)
        ax3_twin.set_ylabel('Momentum change', fontsize=10, color='gray')
        ax3_twin.tick_params(axis='y', labelcolor='gray')
        ax3_twin.legend(loc='upper right', fontsize=8)

    ax3.set_xlabel('Time $t$', fontsize=11)
    ax3.set_ylabel('Normalized quantity', fontsize=11)

    # Calculate errors
    mass_err = np.max(np.abs(mass - mass[0])) / M0 * 100
    momx_err = np.max(np.abs(momx - momx[0]))
    momy_err = np.max(np.abs(momy - momy[0]))

    ax3.set_title(
        f'Conservation Check: mass err={mass_err:.2e}%, |$\\Delta M_x$|={momx_err:.2e}, |$\\Delta M_y$|={momy_err:.2e}',
        fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, T_final])

    # ==================== Title ====================
    fig.suptitle(f'{category} - Sample {sample_idx} (dry_frac = {dry_frac:.2%})',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return {
        'mass_error_percent': mass_err,
        'momx_error_abs': momx_err,
        'momy_error_abs': momy_err,
        'h_min': np.min(h_min),
        'dry_frac': dry_frac
    }


# ============================================================
# Dataset Generation Main Function
# ============================================================

def generate_all_samples():
    """
    Generate all samples with stratified split
    """
    print("=" * 70)
    print("2D Shallow Water Equations Dataset Generator")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Domain: [{Lx} x {Ly}], Grid: {Nx} x {Ny} (computation)")
    print(f"  Saved grid: {Nx // SPACE_DOWNSAMPLE} x {Ny // SPACE_DOWNSAMPLE}")
    print(f"  T_final = {T_final}, FIXED dt = {FIXED_DT}")
    print(f"  Time downsample: {TIME_DOWNSAMPLE}, Space downsample: {SPACE_DOWNSAMPLE}")
    print(f"  Global seed: {GLOBAL_SEED}")
    print("=" * 70)

    # Create directories
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Master RNG
    master_rng = np.random.default_rng(GLOBAL_SEED)

    # Storage
    all_samples = {split: [] for split in ['train', 'val', 'test']}

    total_generated = 0
    total_with_dry = 0

    for category, config in CATEGORY_CONFIG.items():
        print(f"\n{'=' * 50}")
        print(f"Generating {category}: {config['total']} samples")
        print(f"  Split: train={config['train']}, val={config['val']}, test={config['test']}")
        print(f"{'=' * 50}")

        generator = GENERATORS[category]
        category_samples = []

        for i in range(config['total']):
            # Create independent seed for this sample
            sample_seed = master_rng.integers(0, 2 ** 31)
            sample_rng = np.random.default_rng(sample_seed)

            # Generate initial conditions
            h0, mx0, my0, params = generator(sample_rng)

            # Simulate with FIXED time step
            (times, h_hist, mx_hist, my_hist,
             mass_hist, momx_hist, momy_hist,
             h_min_hist, h_max_hist) = simulate_swe(
                h0, mx0, my0, T_final, FIXED_DT, TIME_DOWNSAMPLE
            )

            # Downsample spatially (conservative box averaging)
            h_coarse = downsample_spatial_conservative(h_hist, SPACE_DOWNSAMPLE)
            mx_coarse = downsample_spatial_conservative(mx_hist, SPACE_DOWNSAMPLE)
            my_coarse = downsample_spatial_conservative(my_hist, SPACE_DOWNSAMPLE)

            # Get coarse grid
            x_coarse, y_coarse, dx_coarse, dy_coarse = get_coarse_grid(SPACE_DOWNSAMPLE)

            # Statistics
            dry_frac = params.get('dry_frac', 0.0)
            h_global_min = np.min(h_min_hist)

            # Mass conservation error
            M0 = mass_hist[0]
            max_mass_error = np.max(np.abs(mass_hist - M0)) / M0 * 100 if M0 != 0 else 0

            # Velocity stats (for Case B1, B2)
            eps = 1e-10
            h_safe = np.maximum(h0, eps)
            max_u = np.max(np.abs(mx0 / h_safe))
            max_v = np.max(np.abs(my0 / h_safe))
            max_mx = np.max(np.abs(mx0))
            max_my = np.max(np.abs(my0))

            category_samples.append({
                'category': category,
                'idx': i,
                'seed': sample_seed,
                'h': h_coarse,
                'mx': mx_coarse,
                'my': my_coarse,
                'x': x_coarse,
                'y': y_coarse,
                'times': times,
                'mass': mass_hist,
                'momx': momx_hist,
                'momy': momy_hist,
                'h_min': h_min_hist,
                'h_max': h_max_hist,
                'params': params,
                'h_global_min': h_global_min,
                'max_mass_error': max_mass_error,
                'dry_frac': dry_frac
            })

            if dry_frac > 0:
                total_with_dry += 1

            # Print progress
            print(f"  [{i + 1:3d}/{config['total']}] mass={M0:.4f}, min(h)={h_global_min:.6f}, "
                  f"dry_frac={dry_frac:.2%}, max|mx|={max_mx:.4f}, max|my|={max_my:.4f}")

        # Shuffle and split
        indices = list(range(config['total']))
        master_rng.shuffle(indices)

        train_indices = indices[:config['train']]
        val_indices = indices[config['train']:config['train'] + config['val']]
        test_indices = indices[config['train'] + config['val']:]

        # Assign to splits
        for idx in train_indices:
            sample = category_samples[idx]
            sample['split'] = 'train'
            sample['global_idx'] = len(all_samples['train'])
            all_samples['train'].append(sample)

        for idx in val_indices:
            sample = category_samples[idx]
            sample['split'] = 'val'
            sample['global_idx'] = len(all_samples['val'])
            all_samples['val'].append(sample)

        for idx in test_indices:
            sample = category_samples[idx]
            sample['split'] = 'test'
            sample['global_idx'] = len(all_samples['test'])
            all_samples['test'].append(sample)

        total_generated += config['total']

    print(f"\n{'=' * 70}")
    print(f"Total samples generated: {total_generated}")
    print(f"  With dry regions (h=0): {total_with_dry} ({total_with_dry / total_generated:.1%})")
    print(f"  Train: {len(all_samples['train'])}")
    print(f"  Val: {len(all_samples['val'])}")
    print(f"  Test: {len(all_samples['test'])}")
    print(f"{'=' * 70}")

    return all_samples


def save_all_samples(all_samples):
    """
    Save all samples to HDF5 files
    """
    print("\n" + "=" * 70)
    print("Saving samples to HDF5 files")
    print("=" * 70)

    # Resolution info for filename
    Nx_saved = Nx // SPACE_DOWNSAMPLE
    Ny_saved = Ny // SPACE_DOWNSAMPLE

    split_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}

    for split, samples in all_samples.items():
        print(f"\nSaving {split} set ({len(samples)} samples)...")

        for sample in samples:
            # Filename includes resolution info
            filename = (f"{sample['category']}_sample{sample['idx']:02d}_"
                        f"Nx{Nx_saved}_Ny{Ny_saved}_Nt{len(sample['times'])}_"
                        f"tds{TIME_DOWNSAMPLE}_sds{SPACE_DOWNSAMPLE}.h5")
            filepath = split_dirs[split] / filename

            metadata = {
                'h_global_min': sample['h_global_min'],
                'max_mass_error_percent': sample['max_mass_error'],
                'dry_frac_initial': sample['dry_frac'],
                'seed': sample['seed']
            }

            save_sample_h5(
                filepath,
                sample['h'],
                sample['mx'],
                sample['my'],
                sample['x'],
                sample['y'],
                sample['times'],
                sample['category'],
                sample['idx'],
                sample['params'],
                sample['mass'],
                sample['momx'],
                sample['momy'],
                sample['h_min'],
                sample['h_max'],
                metadata
            )

        print(f"  Saved {len(samples)} files to {split_dirs[split]}")


def visualize_all_samples(all_samples):
    """
    Generate visualization for ALL samples
    """
    print("\n" + "=" * 70)
    print("Generating visualizations for ALL samples")
    print("=" * 70)

    split_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}
    Nx_saved = Nx // SPACE_DOWNSAMPLE
    Ny_saved = Ny // SPACE_DOWNSAMPLE

    all_stats = []

    for split, samples in all_samples.items():
        print(f"\nVisualizing {split} set ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            # Find the h5 file
            filename = (f"{sample['category']}_sample{sample['idx']:02d}_"
                        f"Nx{Nx_saved}_Ny{Ny_saved}_Nt{len(sample['times'])}_"
                        f"tds{TIME_DOWNSAMPLE}_sds{SPACE_DOWNSAMPLE}.h5")
            h5_path = split_dirs[split] / filename

            # Output figure path
            fig_filename = f"{split}_{sample['category']}_sample{sample['idx']:02d}.png"
            fig_path = FIG_DIR / fig_filename

            # Generate visualization
            stats = visualize_sample(h5_path, fig_path)
            stats['split'] = split
            stats['category'] = sample['category']
            stats['sample_idx'] = sample['idx']
            all_stats.append(stats)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed {i + 1}/{len(samples)}")

    print(f"\nGenerated {len(all_stats)} visualization figures in {FIG_DIR}")

    return all_stats


def print_summary(all_samples, all_stats):
    """
    Print final summary
    """
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    print("\n1. Dataset Statistics:")
    print("-" * 50)
    for split in ['train', 'val', 'test']:
        samples = all_samples[split]
        print(f"\n  {split.upper()} ({len(samples)} samples):")

        # Count by category
        cat_counts = {}
        for s in samples:
            cat = s['category']
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count}")

    print("\n2. Dry Region Statistics:")
    print("-" * 50)
    dry_fracs = [s['dry_frac'] for samples in all_samples.values() for s in samples]
    has_dry = sum(1 for d in dry_fracs if d > 0)
    print(f"  Samples with dry regions: {has_dry}/{len(dry_fracs)} ({has_dry / len(dry_fracs):.1%})")
    print(f"  Mean dry fraction (when >0): {np.mean([d for d in dry_fracs if d > 0]):.2%}")
    print(f"  Max dry fraction: {max(dry_fracs):.2%}")

    print("\n3. Conservation Check:")
    print("-" * 50)
    mass_errors = [s['mass_error_percent'] for s in all_stats]
    momx_errors = [s['momx_error_abs'] for s in all_stats]
    momy_errors = [s['momy_error_abs'] for s in all_stats]
    print(f"  Max relative mass error: {max(mass_errors):.2e}%")
    print(f"  Mean relative mass error: {np.mean(mass_errors):.2e}%")
    print(f"  Max |ΔMx|: {max(momx_errors):.2e}")
    print(f"  Max |ΔMy|: {max(momy_errors):.2e}")

    print("\n4. Water Depth Boundedness:")
    print("-" * 50)
    h_mins = [s['h_min'] for s in all_stats]
    print(f"  Global minimum h: {min(h_mins):.6f}")
    if min(h_mins) >= -1e-10:
        print("  ✓ All solutions satisfy h >= 0")
    else:
        violations = sum(1 for h in h_mins if h < -1e-10)
        print(f"  ✗ WARNING: {violations} samples have h < 0!")

    print("\n5. Spatial Downsampling Method:")
    print("-" * 50)
    print(f"  Method: Box Averaging (Coarse Graining)")
    print(f"  Factor: {SPACE_DOWNSAMPLE} (Grid: {Nx}x{Ny} -> {Nx // SPACE_DOWNSAMPLE}x{Ny // SPACE_DOWNSAMPLE})")
    print(f"  ")
    print(f"  This method preserves total conserved quantities:")
    print(f"    sum(field_coarse * dx_coarse * dy_coarse) = sum(field_fine * dx_fine * dy_fine)")
    print(f"  ")
    print(f"  For h (water depth): Box averaging gives the average depth in each coarse cell.")
    print(f"    Total volume is preserved: sum(h_coarse)*A_coarse = sum(h_fine)*A_fine")
    print(f"  ")
    print(f"  For mx, my (momentum): Box averaging gives the average momentum density.")
    print(f"    Total momentum is preserved: sum(mx_coarse)*A_coarse = sum(mx_fine)*A_fine")
    print(f"  ")
    print(f"  CONCLUSION: Box averaging is the CORRECT method for conserved quantities")
    print(f"  because it maintains the integral (total amount) while reducing resolution.")

    print("\n6. Time Stepping:")
    print("-" * 50)
    print(f"  FIXED time step: dt = {FIXED_DT} for ALL samples and ALL time steps")
    print(f"  Time downsample: save every {TIME_DOWNSAMPLE} steps")
    print(f"  Total simulation steps: ~{int(T_final / FIXED_DT)}")
    print(f"  Saved time frames: ~{int(T_final / FIXED_DT / TIME_DOWNSAMPLE)}")

    print("\n7. Output Structure:")
    print("-" * 50)
    print(f"  {BASE_DIR}/")
    print(f"  ├── train/     ({len(all_samples['train'])} .h5 files)")
    print(f"  ├── val/       ({len(all_samples['val'])} .h5 files)")
    print(f"  ├── test/      ({len(all_samples['test'])} .h5 files)")
    print(f"  └── figures/   (visualization .png files)")

    print("\n8. HDF5 File Structure:")
    print("-" * 50)
    print("  - h:    (Nt_saved, Nx_saved, Ny_saved) - water depth evolution")
    print("  - mx:   (Nt_saved, Nx_saved, Ny_saved) - x-momentum evolution")
    print("  - my:   (Nt_saved, Nx_saved, Ny_saved) - y-momentum evolution")
    print("  - x, y: spatial grids")
    print("  - t:    time array")
    print("  - mass, momx, momy: conservation histories (full resolution)")
    print("  - h_min, h_max: water depth range histories")
    print("  - metadata/: simulation parameters")
    print("  - params/: case-specific parameters")

    print("\n9. Initial Condition Types:")
    print("-" * 50)
    print("  CaseA1: Multiple Gaussian bumps/depressions (zero momentum)")
    print("  CaseA2: Random low-frequency sinusoidal perturbations (zero momentum)")
    print("  CaseB1: Multiple Gaussian bumps/depressions with initial velocity field")
    print("  CaseB2: Random low-frequency sinusoidal perturbations with initial velocity field")

    print("\n" + "=" * 70)


def inspect_sample_h5(h5_path):
    """
    Print detailed structure of an HDF5 file
    """
    print(f"\nInspecting: {h5_path}")
    print("-" * 50)

    with h5py.File(h5_path, 'r') as f:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: '{name}'")
                print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                data = obj[:]
                print(f"    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: '{name}'")
                for key, val in obj.attrs.items():
                    print(f"    Attr '{key}': {val}")

        f.visititems(print_item)

        # Also print root attributes
        print("  Root attributes:")
        for key, val in f['metadata'].attrs.items():
            print(f"    '{key}': {val}")


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    # Step 1: Generate all samples
    all_samples = generate_all_samples()

    # Step 2: Save to HDF5
    save_all_samples(all_samples)

    # Step 3: Visualize ALL samples
    all_stats = visualize_all_samples(all_samples)

    # Step 4: Print summary
    print_summary(all_samples, all_stats)

    # Step 5: Inspect one sample file
    sample_files = list(TRAIN_DIR.glob("*.h5"))
    if sample_files:
        inspect_sample_h5(sample_files[0])

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
