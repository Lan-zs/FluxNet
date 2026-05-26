"""
Two-Point Statistics Analysis for Spinodal Decomposition - Parallelized Version
Compares three different time step models (10dt, 100dt, 1000dt) with phase-field baseline.

Optimizations:
1. Multi-GPU parallel experiments (4 GPUs)
2. GPU-accelerated two-point statistics using torch.fft
3. Pre-computed radial indices for faster averaging
4. Parallel baseline phase-field simulations
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import joblib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Add the FluxNet source path
sys.path.insert(0, 'FluxNet/src')

# ==================== Configuration ====================
MODEL_PATHS = {
    '10dt': 'FluxNet/results/spinodal_decomposition/ablation_10dt/FluxNet_D_pf/best_model.pt',
    '100dt': 'FluxNet/results/spinodal_decomposition/ablation_100dt/FluxNet_D_pf/best_model.pt',
    '1000dt': 'FluxNet/results/spinodal_decomposition/ablation_1000dt/FluxNet_D_pf/best_model.pt',
}

MODEL_CONFIGS = {
    '10dt': {'base_channels': 32, 'num_blocks': 4, 'kernel_size': 3, 'neighborhood_size': 3},
    '100dt': {'base_channels': 32, 'num_blocks': 4, 'kernel_size': 5, 'neighborhood_size': 5},
    '1000dt': {'base_channels': 32, 'num_blocks': 6, 'kernel_size': 7, 'neighborhood_size': 9},
}

GRID_SIZE = 1024
C0 = 0.60
NOISE_AMP = 0.05
WARMUP_TIME = 2000
TOTAL_TIME = 102000
TRAINING_LENGTH = 50000
STATS_INTERVAL = 1000
NUM_EXPERIMENTS = 20
BASE_SEED = 666
OUTPUT_DIR = 'FluxNet/results/spinodal_decomposition/two_point_analysis'

PF_DT = 1.0e-2
PF_M = 1.0
PF_K = 3.57e-1
PF_R = 8.314
PF_T = 973.15

GPUS = [0, 1, 2, 3]
NUM_GPUS = len(GPUS)

# Number of parallel workers (one per GPU)
NUM_WORKERS = NUM_GPUS


# ==================== Utility Functions ====================
def setup_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_initial_noise(I, J, c0=0.60, noise_amp=0.05, seed=None):
    """Generate initial concentration field with random noise."""
    if seed is not None:
        np.random.seed(seed)
    random_values = np.random.uniform(0, 1, size=(I, J))
    con = c0 + noise_amp * (0.5 - random_values)
    return con.astype(np.float32)


# ==================== GPU-Accelerated Two-Point Statistics ====================
class RadialAverager:
    """Pre-computed radial averager for fast two-point statistics"""

    def __init__(self, shape, max_radius, device):
        self.device = device
        self.max_radius = max_radius
        h, w = shape

        # Pre-compute radial distance map
        y = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
        x = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
        center_y, center_x = h // 2, w // 2
        r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Pre-compute masks and counts for each radius
        self.masks = []
        self.counts = []
        for radius in range(max_radius):
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            count = mask.sum().item()
            self.masks.append(mask)
            self.counts.append(count if count > 0 else 1)  # Avoid division by zero

        self.radii = np.arange(max_radius)

    def compute_radial_average(self, image_2pt):
        """Compute radial average using pre-computed masks"""
        correlation = torch.zeros(self.max_radius, device=self.device, dtype=torch.float32)
        for i, (mask, count) in enumerate(zip(self.masks, self.counts)):
            if count > 0:
                correlation[i] = image_2pt[mask].sum() / count
        return self.radii, correlation.cpu().numpy()


def autocor_gpu(data_tensor):
    """Compute autocorrelation function using FFT on GPU"""
    m, n = data_tensor.shape
    H = torch.fft.fft2(data_tensor)
    H = H * torch.conj(H)
    H = torch.fft.ifft2(H)
    H = torch.fft.fftshift(H.real)
    return H / (m * n)


def calculate_two_point_statistics_gpu(data, radial_averager, device):
    """Calculate two-point statistics on GPU with pre-computed radial averager"""
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    else:
        data_tensor = data.to(dtype=torch.float32)

    image_2pt = autocor_gpu(data_tensor)
    r, S = radial_averager.compute_radial_average(image_2pt)
    return r, S


def calculate_two_point_error(S1, S2):
    """Calculate MAE between two radial correlation functions"""
    min_len = min(len(S1), len(S2))
    return np.mean(np.abs(S1[:min_len] - S2[:min_len]))


def calculate_phase_fractions(data, threshold_high=0.6, threshold_low=0.6):
    """Calculate phase volume fractions"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    phase1_fraction = np.mean(data >= threshold_high)
    phase2_fraction = np.mean(data < threshold_low)
    return phase1_fraction, phase2_fraction


# ==================== Phase-Field Simulation (GPU) ====================
def run_phase_field_step_gpu(con_gpu, m, n, dt=PF_DT, M=PF_M, k=PF_K, R=PF_R, T=PF_T):
    """Run one step of phase-field simulation on GPU."""
    dx = dy = 1.0
    A0 = 15000.0 + 6.1 * T
    A1 = -7600 + 3.55 * T

    # Apply periodic boundary conditions
    con_gpu[0, 1:n + 1] = con_gpu[m, 1:n + 1]
    con_gpu[m + 1, 1:n + 1] = con_gpu[1, 1:n + 1]
    con_gpu[1:m + 1, 0] = con_gpu[1:m + 1, n]
    con_gpu[1:m + 1, n + 1] = con_gpu[1:m + 1, 1]
    con_gpu[0, 0] = con_gpu[m, n]
    con_gpu[0, n + 1] = con_gpu[m, 1]
    con_gpu[m + 1, 0] = con_gpu[1, n]
    con_gpu[m + 1, n + 1] = con_gpu[1, 1]

    c_interior = con_gpu[1:m + 1, 1:n + 1]
    c_safe = torch.clamp(c_interior, 1e-6, 1 - 1e-6)

    dcon = torch.zeros_like(con_gpu)
    dcon[1:m + 1, 1:n + 1] = (
        R * T * torch.log(c_safe / (1.0 - c_safe)) +
        (1.0 - 2.0 * c_safe) * A0 +
        (-6.0 * c_safe + 6.0 * c_safe ** 2 + 1.0) * A1
    ) / (R * T)

    c1 = con_gpu[2:m + 2, 1:n + 1]
    c2 = con_gpu[0:m, 1:n + 1]
    c3 = con_gpu[1:m + 1, 2:n + 2]
    c4 = con_gpu[1:m + 1, 0:n]
    c5 = con_gpu[1:m + 1, 1:n + 1]

    lap_con = torch.zeros_like(con_gpu)
    lap_con[1:m + 1, 1:n + 1] = (c1 + c2 + c3 + c4 - 4.0 * c5) / (dx * dy)

    dF = torch.zeros_like(con_gpu)
    dF[1:m + 1, 1:n + 1] = dcon[1:m + 1, 1:n + 1] - 2 * k * lap_con[1:m + 1, 1:n + 1]

    dF[0, 1:n + 1] = dF[m, 1:n + 1]
    dF[m + 1, 1:n + 1] = dF[1, 1:n + 1]
    dF[1:m + 1, 0] = dF[1:m + 1, n]
    dF[1:m + 1, n + 1] = dF[1:m + 1, 1]
    dF[0, 0] = dF[m, n]
    dF[0, n + 1] = dF[m, 1]
    dF[m + 1, 0] = dF[1, n]
    dF[m + 1, n + 1] = dF[1, 1]

    F1 = dF[2:m + 2, 1:n + 1]
    F2 = dF[0:m, 1:n + 1]
    F3 = dF[1:m + 1, 2:n + 2]
    F4 = dF[1:m + 1, 0:n]
    F5 = dF[1:m + 1, 1:n + 1]

    lap_dF = (F1 + F2 + F3 + F4 - 4.0 * F5) / (dx * dy)
    con_gpu[1:m + 1, 1:n + 1] += dt * M * lap_dF


def run_phase_field_simulation_gpu(initial_con, num_steps, m, n, device, record_interval=None):
    """Run phase-field simulation on specified GPU."""
    con_gpu = torch.zeros((m + 2, n + 2), device=device, dtype=torch.float32)
    con_gpu[1:m + 1, 1:n + 1] = torch.from_numpy(initial_con).to(device)

    recorded_states = {}

    for step in range(num_steps):
        run_phase_field_step_gpu(con_gpu, m, n)

        if record_interval is not None and (step + 1) % record_interval == 0:
            recorded_states[step + 1] = con_gpu[1:m + 1, 1:n + 1].cpu().numpy().copy()

    final_con = con_gpu[1:m + 1, 1:n + 1].cpu().numpy()
    return final_con, recorded_states


# ==================== Model Loading and Inference ====================
def load_model_to_device(checkpoint_path, model_config, device):
    """Load a FluxNet_D model to specified device"""
    from models.fluxnet_d_2d import FluxNet_D

    model = FluxNet_D(
        in_channels=1,
        base_channels=model_config['base_channels'],
        num_blocks=model_config['num_blocks'],
        kernel_size=model_config['kernel_size'],
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d,
        neighborhood_size=model_config['neighborhood_size']
    )
    model.to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    model.eval()
    return model


def run_model_inference(model, current_phi, device):
    """Run single step of model inference"""
    with torch.no_grad():
        if isinstance(current_phi, np.ndarray):
            input_tensor = torch.from_numpy(current_phi).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            input_tensor = current_phi.unsqueeze(0).unsqueeze(0)
        output = model(input_tensor)
        if isinstance(output, tuple):
            predicted_phi = output[0].squeeze()
        else:
            predicted_phi = output.squeeze()
    return predicted_phi


# ==================== Baseline Worker ====================
def run_baseline_worker(args):
    """Worker function to run one baseline phase-field simulation"""
    seed, gpu_id, m, n, total_time, stats_interval = args
    device = torch.device(f'cuda:{gpu_id}')

    print(f"  [GPU {gpu_id}] Starting baseline simulation with seed {seed}")
    start_time = time.time()

    initial_con = generate_initial_noise(m, n, C0, NOISE_AMP, seed)
    _, states = run_phase_field_simulation_gpu(initial_con, total_time, m, n, device, stats_interval)

    print(f"  [GPU {gpu_id}] Completed in {time.time() - start_time:.1f}s")
    return seed, states


def run_baseline_phase_field_analysis_parallel(seed1, seed2, output_dir):
    """Run two baseline phase-field simulations in parallel on different GPUs"""
    print("=" * 60)
    print("BASELINE PHASE-FIELD ANALYSIS (Parallel)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, 'baseline_data.joblib')

    if os.path.exists(cache_file):
        print(f"Loading cached baseline data from {cache_file}")
        return joblib.load(cache_file)

    m = n = GRID_SIZE

    # Run two simulations in parallel on GPU 0 and GPU 1
    args_list = [
        (seed1, GPUS[0], m, n, TOTAL_TIME, STATS_INTERVAL),
        (seed2, GPUS[1], m, n, TOTAL_TIME, STATS_INTERVAL),
    ]

    print(f"\nRunning 2 baseline simulations in parallel on GPUs {GPUS[0]} and {GPUS[1]}...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_baseline_worker, args_list))

    print(f"Total baseline time: {time.time() - start_time:.1f}s")

    # Organize results
    states1 = results[0][1] if results[0][0] == seed1 else results[1][1]
    states2 = results[1][1] if results[1][0] == seed2 else results[0][1]

    # Compute two-point statistics (on GPU 0)
    print("\nComputing two-point statistics...")
    device = torch.device(f'cuda:{GPUS[0]}')
    radial_averager = RadialAverager((m, n), m // 2, device)

    record_times = list(range(WARMUP_TIME, TOTAL_TIME + 1, STATS_INTERVAL))

    baseline_times = []
    baseline_errors = []
    pf1_phase1_fractions = []
    pf1_phase2_fractions = []
    pf2_phase1_fractions = []
    pf2_phase2_fractions = []
    pf1_radial_data = {}

    key_times_T = {
        '0T': WARMUP_TIME,
        '1T': WARMUP_TIME + TRAINING_LENGTH,
        '1.5T': WARMUP_TIME + int(1.5 * TRAINING_LENGTH),
        '2T': TOTAL_TIME
    }
    radial_comparison_data = {}

    for t in record_times:
        if t < WARMUP_TIME or t not in states1 or t not in states2:
            continue

        data1 = states1[t]
        data2 = states2[t]

        r1, S1 = calculate_two_point_statistics_gpu(data1, radial_averager, device)
        r2, S2 = calculate_two_point_statistics_gpu(data2, radial_averager, device)

        error = calculate_two_point_error(S1, S2)
        p1_1, p2_1 = calculate_phase_fractions(data1)
        p1_2, p2_2 = calculate_phase_fractions(data2)

        baseline_times.append(t)
        baseline_errors.append(error)
        pf1_phase1_fractions.append(p1_1)
        pf1_phase2_fractions.append(p2_1)
        pf2_phase1_fractions.append(p1_2)
        pf2_phase2_fractions.append(p2_2)

        pf1_radial_data[t] = {'r': r1, 'S': S1}

        for key, key_t in key_times_T.items():
            if t == key_t:
                radial_comparison_data[key] = {
                    'time': t, 'r': r1, 'S': S1, 'pf1_data': data1.copy()
                }

    avg_phase1_fractions = [(p1 + p2) / 2 for p1, p2 in zip(pf1_phase1_fractions, pf2_phase1_fractions)]
    avg_phase2_fractions = [(p1 + p2) / 2 for p1, p2 in zip(pf1_phase2_fractions, pf2_phase2_fractions)]

    baseline_data = {
        'times': baseline_times,
        'errors': baseline_errors,
        'pf1_phase1_fractions': pf1_phase1_fractions,
        'pf1_phase2_fractions': pf1_phase2_fractions,
        'pf2_phase1_fractions': pf2_phase1_fractions,
        'pf2_phase2_fractions': pf2_phase2_fractions,
        'avg_phase1_fractions': avg_phase1_fractions,
        'avg_phase2_fractions': avg_phase2_fractions,
        'pf1_radial_data': pf1_radial_data,
        'radial_comparison_data': radial_comparison_data,
        'states1': states1,
        'seed1': seed1,
        'seed2': seed2,
    }

    joblib.dump(baseline_data, cache_file)
    print(f"\nBaseline data saved to {cache_file}")

    return baseline_data


# ==================== ML Experiment Worker ====================
def run_ml_experiment_batch(args):
    """
    Worker function to run a batch of ML experiments on a specific GPU.
    Each worker handles multiple experiments to reduce process overhead.
    """
    worker_id, gpu_id, experiment_indices, experiment_seeds, pf1_radial_data_np, stats_times_list = args

    device = torch.device(f'cuda:{gpu_id}')
    m = n = GRID_SIZE

    print(f"  [Worker {worker_id}, GPU {gpu_id}] Starting {len(experiment_indices)} experiments")

    # Load models once for this worker
    sys.path.insert(0, 'FluxNet/src')
    from models.fluxnet_d_2d import FluxNet_D

    models = {}
    for model_name, path in MODEL_PATHS.items():
        models[model_name] = load_model_to_device(path, MODEL_CONFIGS[model_name], device)

    # Create radial averager
    radial_averager = RadialAverager((m, n), m // 2, device)

    # Convert pf1_radial_data to have numpy arrays (it's passed as dict of dicts)
    pf1_S_data = {t: data['S'] for t, data in pf1_radial_data_np.items()}

    model_dt = {'10dt': 10, '100dt': 100, '1000dt': 1000}

    key_times_T = {
        '0T': WARMUP_TIME,
        '1T': WARMUP_TIME + TRAINING_LENGTH,
        '1.5T': WARMUP_TIME + int(1.5 * TRAINING_LENGTH),
        '2T': TOTAL_TIME
    }

    all_exp_results = []
    first_exp_radial_data = None

    for batch_idx, (exp_idx, exp_seed) in enumerate(zip(experiment_indices, experiment_seeds)):
        start_time = time.time()

        # Generate initial noise
        setup_seed(exp_seed)
        initial_con = generate_initial_noise(m, n, C0, NOISE_AMP, exp_seed)

        # Run phase-field warmup
        warmup_con, _ = run_phase_field_simulation_gpu(initial_con, WARMUP_TIME, m, n, device)

        # Initialize states for all models
        current_states = {
            '10dt': torch.from_numpy(warmup_con).to(device),
            '100dt': torch.from_numpy(warmup_con).to(device),
            '1000dt': torch.from_numpy(warmup_con).to(device)
        }

        exp_results = {
            model_name: {'times': [], 'errors': [], 'phase1_fractions': [], 'phase2_fractions': []}
            for model_name in models.keys()
        }
        exp_radial_data = {model_name: {} for model_name in models.keys()}

        # Record initial state at 0T
        for model_name in models.keys():
            if WARMUP_TIME in pf1_S_data:
                _, S_ml = calculate_two_point_statistics_gpu(current_states[model_name], radial_averager, device)
                S_pf = pf1_S_data[WARMUP_TIME]
                error = calculate_two_point_error(S_ml, S_pf)
                p1, p2 = calculate_phase_fractions(current_states[model_name])

                exp_results[model_name]['times'].append(WARMUP_TIME)
                exp_results[model_name]['errors'].append(error)
                exp_results[model_name]['phase1_fractions'].append(p1)
                exp_results[model_name]['phase2_fractions'].append(p2)

                if batch_idx == 0:  # First experiment in this batch
                    exp_radial_data[model_name]['0T'] = {'S': S_ml.copy()}

        # Rollout from WARMUP_TIME to TOTAL_TIME
        current_times = {name: WARMUP_TIME for name in models.keys()}

        # Create a set for faster lookup
        stats_times_set = set(stats_times_list)

        while any(t < TOTAL_TIME for t in current_times.values()):
            for model_name, model in models.items():
                if current_times[model_name] >= TOTAL_TIME:
                    continue

                dt = model_dt[model_name]

                # Run one model step
                current_states[model_name] = run_model_inference(model, current_states[model_name], device)
                current_times[model_name] += dt

                current_t = current_times[model_name]
                if current_t in stats_times_set and current_t in pf1_S_data:
                    state_np = current_states[model_name].cpu().numpy()
                    _, S_ml = calculate_two_point_statistics_gpu(current_states[model_name], radial_averager, device)
                    S_pf = pf1_S_data[current_t]
                    error = calculate_two_point_error(S_ml, S_pf)
                    p1, p2 = calculate_phase_fractions(state_np)

                    exp_results[model_name]['times'].append(current_t)
                    exp_results[model_name]['errors'].append(error)
                    exp_results[model_name]['phase1_fractions'].append(p1)
                    exp_results[model_name]['phase2_fractions'].append(p2)

                    # Record radial data at key times for first experiment
                    if batch_idx == 0:
                        for key, key_t in key_times_T.items():
                            if current_t == key_t:
                                exp_radial_data[model_name][key] = {'S': S_ml.copy()}

        all_exp_results.append(exp_results)
        if batch_idx == 0:
            first_exp_radial_data = exp_radial_data

        if (batch_idx + 1) % 5 == 0:
            print(f"  [Worker {worker_id}, GPU {gpu_id}] Completed {batch_idx + 1}/{len(experiment_indices)} experiments")

    print(f"  [Worker {worker_id}, GPU {gpu_id}] Finished all experiments")

    return worker_id, all_exp_results, first_exp_radial_data


def run_ml_experiments_parallel(baseline_data, num_experiments, base_seed, output_dir):
    """Run multiple ML inference experiments in parallel across GPUs"""
    print("\n" + "=" * 60)
    print("ML MODEL EXPERIMENTS (Parallel)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, 'ml_experiments_data.joblib')

    if os.path.exists(cache_file):
        print(f"Loading cached ML experiments data from {cache_file}")
        return joblib.load(cache_file)

    # Prepare reference data (convert to serializable format)
    pf1_radial_data = baseline_data['pf1_radial_data']
    pf1_radial_data_serializable = {
        t: {'r': data['r'].tolist() if isinstance(data['r'], np.ndarray) else data['r'],
            'S': data['S'].tolist() if isinstance(data['S'], np.ndarray) else data['S']}
        for t, data in pf1_radial_data.items()
    }
    # Convert back to numpy for the workers
    pf1_radial_data_np = {
        t: {'r': np.array(data['r']), 'S': np.array(data['S'])}
        for t, data in pf1_radial_data_serializable.items()
    }

    stats_times = list(range(WARMUP_TIME, TOTAL_TIME + 1, STATS_INTERVAL))

    # Distribute experiments across workers
    experiments_per_worker = num_experiments // NUM_WORKERS
    remainder = num_experiments % NUM_WORKERS

    worker_args = []
    exp_idx = 0
    for worker_id in range(NUM_WORKERS):
        gpu_id = GPUS[worker_id]
        n_exps = experiments_per_worker + (1 if worker_id < remainder else 0)

        exp_indices = list(range(exp_idx, exp_idx + n_exps))
        exp_seeds = [base_seed + i for i in exp_indices]

        worker_args.append((
            worker_id, gpu_id, exp_indices, exp_seeds, pf1_radial_data_np, stats_times
        ))
        exp_idx += n_exps

    print(f"\nDistributing {num_experiments} experiments across {NUM_WORKERS} workers (GPUs: {GPUS})")
    for i, args in enumerate(worker_args):
        print(f"  Worker {i} (GPU {args[1]}): {len(args[2])} experiments")

    # Run experiments in parallel
    print("\nStarting parallel execution...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_ml_experiment_batch, args) for args in worker_args]
        results = [f.result() for f in futures]

    print(f"\nTotal ML experiments time: {time.time() - start_time:.1f}s")
    print(f"Average time per experiment: {(time.time() - start_time) / num_experiments:.1f}s")

    # Aggregate results from all workers
    print("\nAggregating results...")

    all_experiments = {model_name: [] for model_name in MODEL_PATHS.keys()}
    ml_radial_comparison = None

    for worker_id, exp_results_list, first_exp_radial in results:
        for exp_results in exp_results_list:
            for model_name in MODEL_PATHS.keys():
                all_experiments[model_name].append(exp_results[model_name])

        # Keep radial data from worker 0's first experiment
        if worker_id == 0 and first_exp_radial is not None:
            ml_radial_comparison = first_exp_radial

    # Compute statistics
    aggregated_results = {}
    for model_name in MODEL_PATHS.keys():
        all_exp = all_experiments[model_name]
        times = all_exp[0]['times']

        all_errors = np.array([exp['errors'] for exp in all_exp])
        all_phase1 = np.array([exp['phase1_fractions'] for exp in all_exp])
        all_phase2 = np.array([exp['phase2_fractions'] for exp in all_exp])

        aggregated_results[model_name] = {
            'times': times,
            'errors_mean': np.mean(all_errors, axis=0),
            'errors_std': np.std(all_errors, axis=0),
            'phase1_mean': np.mean(all_phase1, axis=0),
            'phase1_std': np.std(all_phase1, axis=0),
            'phase2_mean': np.mean(all_phase2, axis=0),
            'phase2_std': np.std(all_phase2, axis=0),
            'all_errors': all_errors,
            'all_phase1': all_phase1,
            'all_phase2': all_phase2,
        }

    ml_data = {
        'aggregated_results': aggregated_results,
        'ml_radial_comparison': ml_radial_comparison,
        'num_experiments': num_experiments,
        'base_seed': base_seed,
    }

    joblib.dump(ml_data, cache_file)
    print(f"\nML experiments data saved to {cache_file}")

    return ml_data


def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("TWO-POINT STATISTICS ANALYSIS FOR SPINODAL DECOMPOSITION")
    print("         (Parallelized Multi-GPU Version)")
    print("=" * 60)
    print(f"Available GPUs: {GPUS}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Time range: {WARMUP_TIME} (0T) to {TOTAL_TIME} (2T)")
    print(f"Number of experiments: {NUM_EXPERIMENTS}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Baseline (parallel)
    print("\n\n" + "=" * 60)
    print("STEP 1: Baseline Phase-Field Analysis")
    print("=" * 60)
    baseline_data = run_baseline_phase_field_analysis_parallel(
        seed1=666,
        seed2=624,
        output_dir=OUTPUT_DIR
    )

    # Step 2: ML experiments (parallel)
    print("\n\n" + "=" * 60)
    print("STEP 2: ML Model Experiments")
    print("=" * 60)
    ml_data = run_ml_experiments_parallel(
        baseline_data=baseline_data,
        num_experiments=NUM_EXPERIMENTS,
        base_seed=BASE_SEED,
        output_dir=OUTPUT_DIR
    )

    # Step 3: Save final results
    print("\n\n" + "=" * 60)
    print("STEP 3: Saving Final Results")
    print("=" * 60)

    final_results = {
        'baseline': {
            'times': baseline_data['times'],
            'errors': baseline_data['errors'],
            'avg_phase1_fractions': baseline_data['avg_phase1_fractions'],
            'avg_phase2_fractions': baseline_data['avg_phase2_fractions'],
            'radial_comparison_data': baseline_data['radial_comparison_data'],
        },
        'ml_results': ml_data['aggregated_results'],
        'ml_radial_comparison': ml_data['ml_radial_comparison'],
        'config': {
            'grid_size': GRID_SIZE,
            'warmup_time': WARMUP_TIME,
            'total_time': TOTAL_TIME,
            'training_length': TRAINING_LENGTH,
            'stats_interval': STATS_INTERVAL,
            'num_experiments': NUM_EXPERIMENTS,
            'base_seed': BASE_SEED,
            'model_configs': MODEL_CONFIGS,
            'gpus_used': GPUS,
        }
    }

    final_file = os.path.join(OUTPUT_DIR, 'two_point_analysis_results.joblib')
    joblib.dump(final_results, final_file)
    print(f"Final results saved to {final_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nSummary of results:")
    print(f"  Baseline error range: {min(baseline_data['errors']):.6e} - {max(baseline_data['errors']):.6e}")

    for model_name, results in ml_data['aggregated_results'].items():
        print(f"\n  {model_name} model:")
        print(f"    Final error: {results['errors_mean'][-1]:.6e} ± {results['errors_std'][-1]:.6e}")

    print(f"\nOutput files:")
    print(f"  - Baseline data: {os.path.join(OUTPUT_DIR, 'baseline_data.joblib')}")
    print(f"  - ML experiments: {os.path.join(OUTPUT_DIR, 'ml_experiments_data.joblib')}")
    print(f"  - Final results: {final_file}")


if __name__ == '__main__':
    # Set multiprocessing start method (important for CUDA)
    mp.set_start_method('spawn', force=True)
    main()
