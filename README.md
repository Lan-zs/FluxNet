# FluxNet

Official code for the ICML 2026 paper:

**FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates**

Zishuo Lan, Junjie Li, Lei Wang, Jincheng Wang

[[Paper]](https://openreview.net/forum?id=1KRpajnd6u) [[arXiv]](https://arxiv.org/abs/2602.01941)

## Overview

FluxNet is a neural PDE surrogate that learns capacity-constrained cumulative transport amounts, guaranteeing exact discrete conservation and structural bound preservation by construction. Unlike flux-rate surrogates that inherit CFL constraints, FluxNet enables large-timestep prediction at full spatial resolution via configurable transport neighborhoods.

Key features:
- **Exact conservation** at machine precision via transport update structure
- **Structural bound preservation** through modular transport heads (L, U, D)
- **CFL-free** large-timestep prediction via cumulative transport (no temporal integration)
- **Non-periodic boundary support** via ghost cell extension
- **Modular design**: transport heads are compatible with different backbones (ResNet, FNO)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU
- NumPy, h5py, matplotlib, tqdm

```bash
pip install torch numpy h5py matplotlib tqdm
```

For the spinodal decomposition dataset generator, a CUDA compiler (`nvcc`) is required.

## Project Structure

```
├── dataset/                    # Dataset generation scripts
│   ├── convection_diffusion/
│   ├── shallow_water/
│   ├── spinodal_decomposition/ # CUDA-based phase-field generator
│   └── traffic_flow/
├── experiments/                # Experiment launch scripts
│   ├── convection_diffusion/
│   ├── shallow_water/
│   ├── spinodal_decomposition/
│   └── traffic_flow/
├── src/                        # Core source code
│   ├── models/                 # FluxNet variants and baselines
│   ├── training/               # Training logic and configs
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Visualization utilities
└── results/                    # Auto-created experiment outputs
```

## Usage

### 1. Generate Datasets

```bash
# Convection-diffusion
python dataset/convection_diffusion/dataset.py

# Shallow water
python dataset/shallow_water/dataset.py

# Traffic flow
python dataset/traffic_flow/dataset.py

# Spinodal decomposition (requires CUDA compiler)
nvcc dataset/spinodal_decomposition/phase_field_generator.cu -o pf_generator
./pf_generator
```

### 2. Run Experiments

Single-seed experiments:
```bash
python experiments/convection_diffusion/run_single_seed.py
python experiments/shallow_water/run_single_seed.py
python experiments/traffic_flow/run_single_seed_periodic.py
python experiments/spinodal_decomposition/run_single_seed_100dt.py
```

Multi-seed experiments (5 seeds, as reported in the paper):
```bash
python experiments/convection_diffusion/run_multi_seed.py
python experiments/shallow_water/run_multi_seed.py
python experiments/traffic_flow/run_multi_seed_periodic.py
```

Traffic flow with Dirichlet boundary conditions:
```bash
python experiments/traffic_flow/run_multi_seed_dirichlet_fluxnet_d_t10.py
python experiments/traffic_flow/run_multi_seed_dirichlet_fluxnet_d_t50.py
```

Spinodal decomposition at different temporal strides:
```bash
python experiments/spinodal_decomposition/run_single_seed_10dt.py
python experiments/spinodal_decomposition/run_single_seed_100dt.py
python experiments/spinodal_decomposition/run_single_seed_1000dt.py
```

Results are saved automatically to the `results/` directory.

### 3. Analysis (Spinodal Decomposition)

```bash
python experiments/spinodal_decomposition/analysis/effective_receptive_field_analysis.py
python experiments/spinodal_decomposition/analysis/two_point_statistics_analysis.py
```

## Transport Head Variants

| Head | Constraint | Description |
|------|-----------|-------------|
| N | Conservation only | Signed transport |
| P | Conservation only | Nonneg. transport (softplus) |
| L | u ≥ ℓ | Capacity-limited outflow |
| U | u ≤ u_max | Capacity-limited inflow |
| D | ℓ ≤ u ≤ u_max | Dual-branch with DCL |
| LAP | h ≥ 0 (shallow water) | L-head depth + Advection-Pressure momentum |

## Benchmarks

| Benchmark | Equation | Bound Constraint | Head |
|-----------|----------|-----------------|------|
| Convection–Diffusion (1D) | Advection–diffusion | c ≥ 0 | L |
| Shallow Water (2D) | SWE | h ≥ 0 | LAP |
| Traffic Flow (1D) | LWR | ρ ∈ [0,1] | D |
| Spinodal Decomposition (2D) | Cahn–Hilliard | ϕ ∈ [0,1] | D |

## Citation

```bibtex
@inproceedings{lan2026fluxnet,
  title={FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates},
  author={Lan, Zishuo and Li, Junjie and Wang, Lei and Wang, Jincheng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```