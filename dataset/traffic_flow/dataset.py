#!/usr/bin/env python3
"""
One-Dimensional Traffic Flow LWR Conservation Law Dataset Generator
====================================================================
Model: ∂ρ/∂t + ∂q/∂x = 0
Flux function (Greenshields): q(ρ,x) = vmax(x) · ρ · (1-ρ)
Numerical method: Finite Volume + Rusanov flux
Boundary condition: Periodic (ring road)
Time stepping: FIXED dt (uniform across all simulations)
====================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import h5py
import os
from pathlib import Path
# ============================================================
# Global Parameters
# ============================================================
L = 10.0  # Spatial domain [0, L]
Nx = 256  # Number of grid cells (high resolution for computation)
dx = L / Nx  # Grid spacing
CFL = 0.45  # CFL number (for reference/verification)
# ============================================================
# TIME STEPPING CONFIGURATION (USER-FRIENDLY)
# ============================================================
# KEY DESIGN: Fix dt first, then compute N_TIME_STEPS based on T_final
# This ensures consistent dt across different T_final values
# T_final = 8.0  # Final simulation time (can be 4.0, 8.0, etc.)
T_final = 4.0  # Uncomment for shorter simulations
# Downsampling parameters
TIME_DOWNSAMPLE = 10  # Save every TIME_DOWNSAMPLE steps
SPACE_DOWNSAMPLE = 1  # Spatial coarsening factor (256/2 = 128 points saved)
# ============================================================
# FIXED TIME STEP CALCULATION (CLEAN INTEGER STEPS)
# ============================================================
# Target: For T=8.0 -> 500 steps (50 saved), For T=4.0 -> 250 steps (25 saved)
# This means dt = 8.0/500 = 4.0/250 = 0.016
DT_TARGET = 0.016  # Target fixed time step (same for all simulations)
# Compute N_TIME_STEPS and ensure it's a multiple of TIME_DOWNSAMPLE
N_TIME_STEPS_RAW = T_final / DT_TARGET
N_TIME_STEPS = int(np.ceil(N_TIME_STEPS_RAW / TIME_DOWNSAMPLE)) * TIME_DOWNSAMPLE
# Recompute exact dt to ensure T_final is exactly reached
DT_FIXED = T_final / N_TIME_STEPS
# Number of saved time points (including t=0)
N_SAVED_TIMES = N_TIME_STEPS // TIME_DOWNSAMPLE + 1
# ============================================================
# CFL STABILITY CHECK
# ============================================================
MAX_CHARACTERISTIC_SPEED = 1.0  # Maximum wave speed for LWR model
DT_MAX_CFL = CFL * dx / MAX_CHARACTERISTIC_SPEED  # Maximum stable dt
CFL_EFFECTIVE = DT_FIXED * MAX_CHARACTERISTIC_SPEED / dx
if DT_FIXED > DT_MAX_CFL:
    raise ValueError(f"DT_FIXED={DT_FIXED:.6f} exceeds CFL limit={DT_MAX_CFL:.6f}! "
                     f"Reduce DT_TARGET or increase Nx.")
# ============================================================
# Print Configuration Summary
# ============================================================
print("=" * 60)
print("FIXED TIME STEP CONFIGURATION")
print("=" * 60)
print(f"Spatial: L = {L}, Nx = {Nx}, dx = {dx:.6f}")
print(f"Temporal: T_final = {T_final}")
print("-" * 60)
print(f"DT_TARGET = {DT_TARGET:.6f}")
print(f"DT_FIXED  = {DT_FIXED:.6f} (actual, ensures exact T_final)")
print(f"N_TIME_STEPS = {N_TIME_STEPS} (multiple of {TIME_DOWNSAMPLE})")
print("-" * 60)
print(f"TIME_DOWNSAMPLE = {TIME_DOWNSAMPLE}")
print(f"N_SAVED_TIMES = {N_SAVED_TIMES} (including t=0)")
print(f"Saved dt = {DT_FIXED * TIME_DOWNSAMPLE:.6f}")
print("-" * 60)
print(f"CFL check: DT_MAX_CFL = {DT_MAX_CFL:.6f}")
print(f"CFL effective = {CFL_EFFECTIVE:.4f} (must be < {CFL})")
print(f"Status: {'✓ STABLE' if CFL_EFFECTIVE < CFL else '✗ UNSTABLE'}")
print("=" * 60)
# Grid centers (high resolution)
x_centers = np.linspace(dx / 2, L - dx / 2, Nx)

# Output directories
BASE_DIR = Path("traffic_flow_dataset")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
FIG_DIR = BASE_DIR / "figures"

# Global random seed
GLOBAL_SEED = 42

# Dataset split configuration
CATEGORY_CONFIG = {
    'Case1':   {'total': 38, 'train': 15, 'val': 8, 'test': 15},
    'Case2A':  {'total': 37, 'train': 15, 'val': 7, 'test': 15},
    'Case2B':  {'total': 38, 'train': 15, 'val': 8, 'test': 15},
    'Case3p':  {'total': 37, 'train': 15, 'val': 7, 'test': 15},  # shock forward
    'Case3m':  {'total': 38, 'train': 15, 'val': 8, 'test': 15},  # shock backward
    'Case3_0': {'total': 25, 'train': 10, 'val': 5, 'test': 10},  # stationary shock
    'Case4':   {'total': 37, 'train': 15, 'val': 7, 'test': 15},
}

# CATEGORY_CONFIG = {
#     'Case1':   {'total': 15, 'train': 0, 'val': 0, 'test': 15},
#     'Case2A':  {'total': 15, 'train': 0, 'val': 0, 'test': 15},
#     'Case2B':  {'total': 15, 'train': 0, 'val': 0, 'test': 15},
#     'Case3p':  {'total': 15, 'train': 0, 'val': 0, 'test': 15},  # shock forward
#     'Case3m':  {'total': 15, 'train': 0, 'val': 0, 'test': 15},  # shock backward
#     'Case3_0': {'total': 10, 'train': 0, 'val': 0, 'test': 10},  # stationary shock
#     'Case4':   {'total': 15, 'train': 0, 'val': 0, 'test': 15},
# }

# ============================================================
# Traffic colormap (green=free flow, red=congestion)
# ============================================================
def create_traffic_cmap():
    """Create traffic light style colormap: green(low)->yellow->red(high)"""
    colors = ['#2ECC40', '#FFDC00', '#FF4136']
    return LinearSegmentedColormap.from_list('traffic', colors, N=256)


TRAFFIC_CMAP = create_traffic_cmap()


# ============================================================
# LWR Model Physics Functions
# ============================================================
def flux_greenshields(rho, vmax):
    """Greenshields flux: q(rho) = vmax * rho * (1 - rho)"""
    return vmax * rho * (1.0 - rho)


def flux_derivative(rho, vmax):
    """Characteristic speed: dq/drho = vmax * (1 - 2*rho)"""
    return vmax * (1.0 - 2.0 * rho)


# ============================================================
# Numerical Solver (Rusanov/Local Lax-Friedrichs) - FIXED DT VERSION
# ============================================================
def time_step_vectorized(rho, vmax, dx_val, dt):
    """Vectorized conservative time stepping with periodic BC"""
    N = len(rho)

    # Periodic extension
    rho_ext = np.concatenate([[rho[-1]], rho, [rho[0]]])
    vmax_ext = np.concatenate([[vmax[-1]], vmax, [vmax[0]]])

    # Interface values
    rho_L = rho_ext[:-1]
    rho_R = rho_ext[1:]
    vmax_L = vmax_ext[:-1]
    vmax_R = vmax_ext[1:]

    # Interface vmax: arithmetic average
    vmax_face = 0.5 * (vmax_L + vmax_R)

    # Fluxes at cell interfaces
    f_L = flux_greenshields(rho_L, vmax_face)
    f_R = flux_greenshields(rho_R, vmax_face)

    # Local wave speeds
    alpha_L = np.abs(flux_derivative(rho_L, vmax_face))
    alpha_R = np.abs(flux_derivative(rho_R, vmax_face))
    alpha = np.maximum(alpha_L, alpha_R)
    alpha = np.maximum(alpha, vmax_face)

    # Rusanov flux
    F_all = 0.5 * (f_L + f_R) - 0.5 * alpha * (rho_R - rho_L)

    # Conservative update
    rho_new = rho - (dt / dx_val) * (F_all[1:N + 1] - F_all[0:N])

    return rho_new


def simulate_lwr_fixed_dt(rho0, vmax, T_end, dx_val, dt_fixed, time_downsample=1):
    """
    Simulate LWR equation evolution with FIXED time step

    Parameters:
        rho0: initial density field
        vmax: maximum velocity field (spatially varying)
        T_end: final simulation time
        dx_val: spatial grid spacing
        dt_fixed: FIXED time step (same for all steps and all simulations)
        time_downsample: save every N steps

    Returns:
        times: recorded time array (uniformly spaced)
        rho_history: density field history (n_records, Nx)
        mass_history: total mass history
    """
    rho = rho0.copy()
    t = 0.0

    times = [0.0]
    rho_history = [rho.copy()]
    mass_history = [np.sum(rho) * dx_val]

    step_count = 0

    # Calculate total number of steps
    n_total_steps = int(np.round(T_end / dt_fixed))

    for step in range(n_total_steps):
        # Use FIXED dt for all steps
        rho = time_step_vectorized(rho, vmax, dx_val, dt_fixed)
        t += dt_fixed
        step_count += 1

        # Record at specified interval
        if step_count % time_downsample == 0:
            times.append(t)
            rho_history.append(rho.copy())
            mass_history.append(np.sum(rho) * dx_val)

    # Ensure final state is recorded
    if step_count % time_downsample != 0:
        times.append(t)
        rho_history.append(rho.copy())
        mass_history.append(np.sum(rho) * dx_val)

    return np.array(times), np.array(rho_history), np.array(mass_history)


# ============================================================
# Spatial Downsampling (Box Averaging for Conservation)
# ============================================================
def downsample_spatial(rho_history, vmax, factor):
    """
    Downsample spatial resolution using box averaging (coarse graining)
    """
    Nx_fine = rho_history.shape[1]
    Nx_coarse = Nx_fine // factor

    # Reshape and average
    rho_reshaped = rho_history.reshape(rho_history.shape[0], Nx_coarse, factor)
    rho_coarse = np.mean(rho_reshaped, axis=2)

    vmax_reshaped = vmax.reshape(Nx_coarse, factor)
    vmax_coarse = np.mean(vmax_reshaped, axis=1)

    # Coarse grid centers
    dx_coarse = L / Nx_coarse
    x_coarse = np.linspace(dx_coarse / 2, L - dx_coarse / 2, Nx_coarse)

    return rho_coarse, vmax_coarse, x_coarse


# ============================================================
# Periodic Distance Helper
# ============================================================
def periodic_distance(x, x0, L_domain):
    """Compute shortest distance considering periodic BC"""
    d = np.abs(x - x0)
    return np.minimum(d, L_domain - d)


def periodic_window_tanh(x, x_center, width, eps, L_domain):
    """
    Periodic smooth window function using tanh
    """
    x1 = x_center - width / 2
    x2 = x_center + width / 2

    window = np.zeros_like(x)

    for i, xi in enumerate(x):
        val_direct = 0.5 * (np.tanh((xi - x1) / eps) - np.tanh((xi - x2) / eps))
        val_wrap_pos = 0.5 * (np.tanh((xi - x1 - L_domain) / eps) - np.tanh((xi - x2 - L_domain) / eps))
        val_wrap_neg = 0.5 * (np.tanh((xi - x1 + L_domain) / eps) - np.tanh((xi - x2 + L_domain) / eps))
        window[i] = max(val_direct, val_wrap_pos, val_wrap_neg, 0)

    return np.clip(window, 0, 1)


# ============================================================
# Initial Condition Generators (Randomized)
# ============================================================
def generate_case1_traffic_jam(rng, x):
    """Case 1: Traffic Jam Ahead (Ramp Structure)"""
    x_jam_start = rng.uniform(0, L)
    ramp_len = rng.uniform(0.1 * L, 0.4 * L)
    plateau_len = rng.uniform(0.02 * L, 0.15 * L)
    rho_low = rng.uniform(0.05, 0.30)
    rho_high = rng.uniform(0.65, 0.95)

    rho = np.zeros_like(x)

    for i, xi in enumerate(x):
        xi_shifted = (xi - x_jam_start) % L

        if xi_shifted < ramp_len:
            rho[i] = rho_low + (rho_high - rho_low) * (xi_shifted / ramp_len)
        elif xi_shifted < ramp_len + plateau_len:
            rho[i] = rho_high
        else:
            rho[i] = rho_low

    vmax = np.ones_like(x)

    params = {
        'x_jam_start': x_jam_start,
        'ramp_len': ramp_len,
        'plateau_len': plateau_len,
        'rho_low': rho_low,
        'rho_high': rho_high
    }

    return rho, vmax, "Case1: Traffic Jam Ahead", params


def generate_random_sinusoidal_ic(rng, x, rho0_range=(0.15, 0.45), M_range=(1, 3),
                                  A_range=(0.01, 0.08)):
    """Generate random sinusoidal superposition initial condition"""
    rho0 = rng.uniform(*rho0_range)
    M = rng.integers(M_range[0], M_range[1] + 1)

    rho = np.ones_like(x) * rho0

    for m in range(1, M + 1):
        A_m = rng.uniform(*A_range)
        phi_m = rng.uniform(0, 2 * np.pi)
        rho += A_m * np.sin(2 * np.pi * m * x / L + phi_m)

    rho = np.clip(rho, 0.02, 0.98)

    return rho


def generate_case2a_speed_limit(rng, x):
    """Case 2A: Speed Limit Zone (Poor Road Condition)"""
    r = rng.uniform(0.3, 0.9)
    x_c = rng.uniform(0, L)
    w = rng.uniform(0.05 * L, 0.25 * L)
    k = rng.uniform(2, 6)
    eps = k * dx

    window = periodic_window_tanh(x, x_c, w, eps, L)
    vmax = 1.0 - (1.0 - r) * window

    rho = generate_random_sinusoidal_ic(rng, x)

    params = {
        'r': r,
        'x_c': x_c,
        'w': w,
        'eps': eps,
        'vmax_min': r
    }

    return rho, vmax, "Case2A: Speed Limit Zone", params


def generate_case2b_red_light(rng, x):
    """Case 2B: Red Light (Complete Stop)"""
    x_light = rng.uniform(0, L)
    width_stop = rng.uniform(0.02 * L, 0.06 * L)

    vmax = np.ones_like(x)

    for i, xi in enumerate(x):
        dist = periodic_distance(xi, x_light, L)
        if dist < width_stop / 2:
            vmax[i] = 0.0

    rho = generate_random_sinusoidal_ic(rng, x)

    params = {
        'x_light': x_light,
        'width_stop': width_stop
    }

    return rho, vmax, "Case2B: Red Light Stop", params


def generate_case3_shock_forward(rng, x):
    """Case 3+: Riemann Problem - Forward Shock"""
    rho_L = rng.uniform(0.10, 0.30)
    rho_R_max = min(0.70, 1.0 - rho_L - 0.05)
    rho_R = rng.uniform(0.45, rho_R_max)
    x0 = rng.uniform(0, L)

    rho = np.zeros_like(x)
    for i, xi in enumerate(x):
        xi_shifted = (xi - x0) % L
        if xi_shifted < L / 2:
            rho[i] = rho_R
        else:
            rho[i] = rho_L

    vmax = np.ones_like(x)

    shock_speed = 1.0 - (rho_L + rho_R)
    params = {
        'rho_L': rho_L,
        'rho_R': rho_R,
        'x0': x0,
        'shock_speed': shock_speed
    }

    return rho, vmax, "Case3+: Shock Forward", params


def generate_case3_shock_backward(rng, x):
    """Case 3-: Riemann Problem - Backward Shock"""
    rho_L = rng.uniform(0.35, 0.55)
    rho_R_min = max(0.65, 1.0 - rho_L + 0.05)
    rho_R = rng.uniform(rho_R_min, 0.90)
    x0 = rng.uniform(0, L)

    rho = np.zeros_like(x)
    for i, xi in enumerate(x):
        xi_shifted = (xi - x0) % L
        if xi_shifted < L / 2:
            rho[i] = rho_R
        else:
            rho[i] = rho_L

    vmax = np.ones_like(x)

    shock_speed = 1.0 - (rho_L + rho_R)
    params = {
        'rho_L': rho_L,
        'rho_R': rho_R,
        'x0': x0,
        'shock_speed': shock_speed
    }

    return rho, vmax, "Case3-: Shock Backward", params


def generate_case3_shock_stationary(rng, x):
    """Case 3_0: Riemann Problem - Stationary Shock"""
    rho_L = rng.uniform(0.15, 0.45)
    perturbation = rng.uniform(-0.01, 0.01)
    rho_R = 1.0 - rho_L + perturbation

    if rho_R <= rho_L:
        rho_R = rho_L + 0.1

    rho_R = np.clip(rho_R, rho_L + 0.05, 0.95)

    x0 = rng.uniform(0, L)

    rho = np.zeros_like(x)
    for i, xi in enumerate(x):
        xi_shifted = (xi - x0) % L
        if xi_shifted < L / 2:
            rho[i] = rho_R
        else:
            rho[i] = rho_L

    vmax = np.ones_like(x)

    shock_speed = 1.0 - (rho_L + rho_R)
    params = {
        'rho_L': rho_L,
        'rho_R': rho_R,
        'x0': x0,
        'shock_speed': shock_speed
    }

    return rho, vmax, "Case3_0: Stationary Shock", params


def generate_case4_rarefaction(rng, x):
    """Case 4: Rarefaction Wave"""
    rho_L = rng.uniform(0.55, 0.90)
    rho_R_max = min(0.45, rho_L - 0.15)
    rho_R = rng.uniform(0.05, rho_R_max)
    x0 = rng.uniform(0, L)

    rho = np.zeros_like(x)
    for i, xi in enumerate(x):
        xi_shifted = (xi - x0) % L
        if xi_shifted < L / 2:
            rho[i] = rho_R
        else:
            rho[i] = rho_L

    vmax = np.ones_like(x)

    char_L = 1.0 - 2.0 * rho_L
    char_R = 1.0 - 2.0 * rho_R
    params = {
        'rho_L': rho_L,
        'rho_R': rho_R,
        'x0': x0,
        'char_speed_L': char_L,
        'char_speed_R': char_R
    }

    return rho, vmax, "Case4: Rarefaction Wave", params


# ============================================================
# Generator Mapping
# ============================================================
GENERATORS = {
    'Case1': generate_case1_traffic_jam,
    'Case2A': generate_case2a_speed_limit,
    'Case2B': generate_case2b_red_light,
    'Case3p': generate_case3_shock_forward,
    'Case3m': generate_case3_shock_backward,
    'Case3_0': generate_case3_shock_stationary,
    'Case4': generate_case4_rarefaction,
}


# ============================================================
# Data Saving Functions
# ============================================================
def save_sample_h5(filepath, rho_data, vmax_data, x_grid, times, category,
                   sample_idx, params, mass_history, metadata):
    """Save a single sample to HDF5 file"""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('rho', data=rho_data.astype(np.float32))
        f.create_dataset('vmax', data=vmax_data.astype(np.float32))
        f.create_dataset('x', data=x_grid.astype(np.float32))
        f.create_dataset('t', data=times.astype(np.float32))
        f.create_dataset('mass', data=mass_history.astype(np.float32))

        meta = f.create_group('metadata')
        meta.attrs['L'] = L
        meta.attrs['Nx_original'] = Nx
        meta.attrs['Nx_saved'] = len(x_grid)
        meta.attrs['dx_saved'] = x_grid[1] - x_grid[0] if len(x_grid) > 1 else L / len(x_grid)
        meta.attrs['T_final'] = T_final
        meta.attrs['Nt_saved'] = len(times)
        meta.attrs['CFL'] = CFL
        meta.attrs['dt_fixed'] = DT_FIXED  # NEW: record fixed dt
        meta.attrs['n_total_steps'] = N_TIME_STEPS  # NEW: record total steps
        meta.attrs['category'] = category
        meta.attrs['sample_idx'] = sample_idx
        meta.attrs['time_downsample'] = TIME_DOWNSAMPLE
        meta.attrs['space_downsample'] = SPACE_DOWNSAMPLE

        params_grp = f.create_group('params')
        for key, val in params.items():
            params_grp.attrs[key] = val

        for key, val in metadata.items():
            meta.attrs[key] = val


# ============================================================
# Visualization Functions (Combined 4-Panel Figure)
# ============================================================
def visualize_sample(h5_filepath, save_path):
    """Generate a combined 4-panel visualization figure"""
    with h5py.File(h5_filepath, 'r') as f:
        rho_data = f['rho'][:]
        vmax_data = f['vmax'][:]
        x_grid = f['x'][:]
        times = f['t'][:]
        mass = f['mass'][:]
        category = f['metadata'].attrs['category']
        sample_idx = f['metadata'].attrs['sample_idx']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{category} - Sample {sample_idx}', fontsize=14, fontweight='bold')

    # Panel 1: Spatiotemporal Evolution
    ax1 = axes[0, 0]
    T_mesh, X_mesh = np.meshgrid(times, x_grid)
    im1 = ax1.pcolormesh(X_mesh, T_mesh, rho_data.T, cmap=TRAFFIC_CMAP,
                         vmin=0, vmax=1, shading='auto')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(r'Density $\rho$', fontsize=11)
    ax1.set_xlabel('Position $x$', fontsize=11)
    ax1.set_ylabel('Time $t$', fontsize=11)
    ax1.set_title('Spatiotemporal Evolution', fontsize=12)
    ax1.annotate('', xy=(x_grid[-1] * 0.9, times[0] + (times[-1] - times[0]) * 0.05),
                 xytext=(x_grid[-1] * 0.7, times[0] + (times[-1] - times[0]) * 0.05),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax1.text(x_grid[-1] * 0.8, times[0] + (times[-1] - times[0]) * 0.1,
             'Flow', fontsize=9, color='white', ha='center')

    # Panel 2: Density Profiles
    ax2 = axes[0, 1]
    n_curves = min(6, len(times))
    time_indices = np.linspace(0, len(times) - 1, n_curves, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_curves))
    for idx, color in zip(time_indices, colors):
        ax2.plot(x_grid, rho_data[idx], color=color, linewidth=1.5,
                 label=f't = {times[idx]:.2f}')
    ax2.set_xlabel('Position $x$', fontsize=11)
    ax2.set_ylabel(r'Density $\rho$', fontsize=11)
    ax2.set_title('Density Profiles at Different Times', fontsize=12)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('', xy=(L * 0.95, 0.95), xytext=(L * 0.8, 0.95),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(L * 0.875, 0.98, 'Flow', fontsize=9, color='blue', ha='center')

    # Panel 3: vmax Profile
    ax3 = axes[1, 0]
    ax3.plot(x_grid, vmax_data, 'b-', linewidth=2, label=r'$v_{max}(x)$')
    ax3.fill_between(x_grid, 0, vmax_data, alpha=0.3, color='blue')
    ax3.set_xlabel('Position $x$', fontsize=11)
    ax3.set_ylabel(r'Maximum Velocity $v_{max}$', fontsize=11)
    ax3.set_title('Spatial Maximum Velocity Field', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, L])
    ax3.set_ylim([0, 1.1])
    ax3.legend(loc='best', fontsize=10)
    if np.min(vmax_data) < 0.9:
        ax3.axhline(y=np.min(vmax_data), color='red', linestyle='--',
                    alpha=0.7, label=f'Min = {np.min(vmax_data):.3f}')
        ax3.legend(loc='best', fontsize=10)

    # Panel 4: Mass Conservation
    ax4 = axes[1, 1]
    M0 = mass[0]
    ax4.plot(times, mass, 'b-', linewidth=1.5, label='$M(t)$')
    ax4.axhline(y=M0, color='r', linestyle='--', linewidth=1.5,
                label=f'$M(0)$ = {M0:.6f}')
    ax4.set_xlabel('Time $t$', fontsize=11)
    ax4.set_ylabel('Total Mass $M(t) = \sum \\rho_i \cdot \Delta x$', fontsize=11)
    ax4.set_title('Mass Conservation Check', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4_twin = ax4.twinx()
    rel_error = np.abs(mass - M0) / (np.abs(M0) + 1e-16) * 100
    rel_error_safe = np.maximum(rel_error, 1e-16)
    ax4_twin.semilogy(times, rel_error_safe, 'g--', linewidth=1.5, alpha=0.7)
    ax4_twin.set_ylabel('Relative Error (%)', fontsize=11, color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    max_rel_error = np.max(rel_error)
    ax4.text(0.02, 0.02, f'Max relative error: {max_rel_error:.2e}%',
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return max_rel_error


# ============================================================
# Dataset Generation Main Function
# ============================================================
def generate_all_samples():
    """Generate all samples with stratified split using FIXED dt"""
    print("=" * 70)
    print("Traffic Flow LWR Dataset Generator (FIXED DT VERSION)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Domain: [0, {L}], Nx = {Nx} (computation), "
          f"Nx_saved = {Nx // SPACE_DOWNSAMPLE} (after downsampling)")
    print(f"  T_final = {T_final}, CFL = {CFL}")
    print(f"  *** FIXED dt = {DT_FIXED:.6f} (all simulations) ***")
    print(f"  *** Total time steps = {N_TIME_STEPS} ***")
    print(f"  Time downsample factor: {TIME_DOWNSAMPLE}")
    print(f"  Space downsample factor: {SPACE_DOWNSAMPLE} (using box averaging)")
    print(f"  Global seed: {GLOBAL_SEED}")
    print("=" * 70)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(GLOBAL_SEED)

    all_samples = {split: [] for split in ['train', 'val', 'test']}

    total_generated = 0

    for category, config in CATEGORY_CONFIG.items():
        print(f"\n{'=' * 50}")
        print(f"Generating {category}: {config['total']} samples")
        print(f"  Split: train={config['train']}, val={config['val']}, test={config['test']}")
        print(f"{'=' * 50}")

        generator = GENERATORS[category]
        category_samples = []

        for i in range(config['total']):
            rho0, vmax, description, params = generator(master_rng, x_centers)

            # *** USE FIXED DT SIMULATION ***
            times, rho_history, mass_history = simulate_lwr_fixed_dt(
                rho0, vmax, T_final, dx, DT_FIXED, time_downsample=TIME_DOWNSAMPLE
            )

            rho_coarse, vmax_coarse, x_coarse = downsample_spatial(
                rho_history, vmax, SPACE_DOWNSAMPLE
            )

            rho_min, rho_max = np.min(rho_history), np.max(rho_history)
            M0 = mass_history[0]
            max_mass_error = np.max(np.abs(mass_history - M0)) / M0 * 100

            # Verify uniform time spacing
            dt_recorded = np.diff(times)
            dt_expected = DT_FIXED * TIME_DOWNSAMPLE
            dt_variation = np.std(dt_recorded) / np.mean(dt_recorded) if len(dt_recorded) > 1 else 0

            category_samples.append({
                'category': category,
                'idx': i,
                'rho': rho_coarse,
                'vmax': vmax_coarse,
                'x': x_coarse,
                'times': times,
                'mass': mass_history,
                'params': params,
                'description': description,
                'rho_range': (rho_min, rho_max),
                'max_mass_error': max_mass_error,
                'dt_variation': dt_variation
            })

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Generated {i + 1}/{config['total']}: "
                      f"rho in [{rho_min:.4f}, {rho_max:.4f}], "
                      f"mass error: {max_mass_error:.2e}%, "
                      f"dt variation: {dt_variation:.2e}")

        indices = list(range(config['total']))
        master_rng.shuffle(indices)

        train_indices = indices[:config['train']]
        val_indices = indices[config['train']:config['train'] + config['val']]
        test_indices = indices[config['train'] + config['val']:]

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
    print(f"  Train: {len(all_samples['train'])}")
    print(f"  Val: {len(all_samples['val'])}")
    print(f"  Test: {len(all_samples['test'])}")
    print(f"{'=' * 70}")

    return all_samples


def save_all_samples(all_samples):
    """Save all samples to HDF5 files"""
    print("\n" + "=" * 70)
    print("Saving samples to HDF5 files")
    print("=" * 70)

    Nx_saved = Nx // SPACE_DOWNSAMPLE
    split_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}

    for split, samples in all_samples.items():
        print(f"\nSaving {split} set ({len(samples)} samples)...")

        for sample in samples:
            filename = (f"{sample['category']}_sample{sample['idx']:02d}_"
                        f"Nx{Nx_saved}_Nt{len(sample['times'])}.h5")
            filepath = split_dirs[split] / filename

            metadata = {
                'rho_min': sample['rho_range'][0],
                'rho_max': sample['rho_range'][1],
                'max_mass_error_percent': sample['max_mass_error'],
                'description': sample['description'],
                'dt_variation': sample['dt_variation']  # Should be ~0
            }

            save_sample_h5(
                filepath,
                sample['rho'],
                sample['vmax'],
                sample['x'],
                sample['times'],
                sample['category'],
                sample['idx'],
                sample['params'],
                sample['mass'],
                metadata
            )

        print(f"  Saved {len(samples)} files to {split_dirs[split]}")


def visualize_all_samples(all_samples):
    """Generate visualization for all samples"""
    print("\n" + "=" * 70)
    print("Generating visualizations")
    print("=" * 70)

    split_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}
    Nx_saved = Nx // SPACE_DOWNSAMPLE

    max_errors = []

    for split, samples in all_samples.items():
        print(f"\nVisualizing {split} set ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            filename = (f"{sample['category']}_sample{sample['idx']:02d}_"
                        f"Nx{Nx_saved}_Nt{len(sample['times'])}.h5")
            h5_path = split_dirs[split] / filename

            fig_filename = f"{split}_{sample['category']}_sample{sample['idx']:02d}.png"
            fig_path = FIG_DIR / fig_filename

            max_err = visualize_sample(h5_path, fig_path)
            max_errors.append((split, sample['category'], sample['idx'], max_err))

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed {i + 1}/{len(samples)}")

    print(f"\nGenerated {len(max_errors)} visualization figures in {FIG_DIR}")

    return max_errors


def print_summary(all_samples, max_errors):
    """Print final summary"""
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE (FIXED DT VERSION)")
    print("=" * 70)

    print("\n1. FIXED TIME STEP VERIFICATION:")
    print("-" * 40)
    print(f"  Fixed dt = {DT_FIXED:.6f}")
    print(f"  Total time steps = {N_TIME_STEPS}")
    print(f"  Recorded dt = {DT_FIXED * TIME_DOWNSAMPLE:.6f} (after downsampling)")

    # Verify all samples have uniform time spacing
    dt_variations = [s['dt_variation'] for samples in all_samples.values() for s in samples]
    print(f"  Max dt variation across all samples: {max(dt_variations):.2e}")
    if max(dt_variations) < 1e-10:
        print("  ✓ All samples have UNIFORM time spacing!")
    else:
        print("  ✗ WARNING: Some samples have non-uniform time spacing!")

    print("\n2. Dataset Statistics:")
    print("-" * 40)
    for split in ['train', 'val', 'test']:
        samples = all_samples[split]
        print(f"\n  {split.upper()} ({len(samples)} samples):")

        cat_counts = {}
        for s in samples:
            cat = s['category']
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count}")

    print("\n3. Conservation Check:")
    print("-" * 40)
    errors_array = np.array([e[3] for e in max_errors])
    print(f"  Max relative mass error across all samples: {np.max(errors_array):.2e}%")
    print(f"  Mean relative mass error: {np.mean(errors_array):.2e}%")
    print(f"  Samples with error > 1e-10%: {np.sum(errors_array > 1e-10)}")

    print("\n4. Boundedness Check:")
    print("-" * 40)
    all_rho_mins = [s['rho_range'][0] for samples in all_samples.values() for s in samples]
    all_rho_maxs = [s['rho_range'][1] for samples in all_samples.values() for s in samples]
    print(f"  Global rho minimum: {min(all_rho_mins):.6f}")
    print(f"  Global rho maximum: {max(all_rho_maxs):.6f}")

    if min(all_rho_mins) >= -1e-10 and max(all_rho_maxs) <= 1 + 1e-10:
        print("  ✓ All solutions bounded in [0, 1]")
    else:
        print("  ✗ WARNING: Some solutions exceed [0, 1] bounds!")

    print("\n5. Output Structure:")
    print("-" * 40)
    print(f"  {BASE_DIR}/")
    print(f"  ├── train/     ({len(all_samples['train'])} .h5 files)")
    print(f"  ├── val/       ({len(all_samples['val'])} .h5 files)")
    print(f"  ├── test/      ({len(all_samples['test'])} .h5 files)")
    print(f"  └── figures/   (visualization .png files)")

    print("\n6. HDF5 File Structure:")
    print("-" * 40)
    print("  - rho: (Nt_saved, Nx_saved) - density evolution")
    print("  - vmax: (Nx_saved,) - maximum velocity field")
    print("  - x: (Nx_saved,) - spatial grid")
    print("  - t: (Nt_saved,) - time array (UNIFORMLY SPACED)")
    print("  - mass: (Nt_saved,) - total mass history")
    print("  - metadata/: simulation parameters")
    print("    - dt_fixed: fixed time step used")
    print("    - n_total_steps: total number of time steps")
    print("  - params/: case-specific parameters")

    print("\n" + "=" * 70)


def inspect_sample_h5(h5_path):
    """Print detailed structure of an HDF5 file"""
    print(f"\nInspecting: {h5_path}")
    print("-" * 50)

    with h5py.File(h5_path, 'r') as f:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: '{name}'")
                print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                data = obj[:]
                print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
                if name == 't':
                    dt_recorded = np.diff(data)
                    print(f"    Time step (recorded): {np.mean(dt_recorded):.6f} ± {np.std(dt_recorded):.2e}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: '{name}'")
                for key, val in obj.attrs.items():
                    print(f"    Attr '{key}': {val}")

        f.visititems(print_item)


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    # Step 1: Generate all samples with fixed dt
    all_samples = generate_all_samples()

    # Step 2: Save to HDF5
    save_all_samples(all_samples)

    # Step 3: Visualize all samples
    max_errors = visualize_all_samples(all_samples)

    # Step 4: Print summary
    print_summary(all_samples, max_errors)

    # Step 5: Inspect one sample file
    sample_files = list(TEST_DIR.glob("*.h5"))
    if sample_files:
        inspect_sample_h5(sample_files[0])
