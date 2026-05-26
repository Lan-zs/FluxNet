"""
Shallow Water Equation Specialized Evaluator

Features:
1. Per-field statistics (h, mx, my separately)
2. Water depth violation rate (h < 0)
3. Mass conservation check for h field
4. Case classification visualization (A1, A2, B1, B2)
5. Specialized plots for shallow water (height colormap, velocity vectors)
"""

import os
import json
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class ShallowWaterEvaluator:
    """
    Specialized evaluator for shallow water equations

    Args:
        model: PyTorch model
        device: Computation device
        ndt: Time step interval for predictions
    """

    def __init__(self, model, device, ndt: int = 1):
        self.model = model
        self.device = device
        self.ndt = ndt
        self.model.eval()

    def evaluate_trajectory(
        self,
        h5_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        Evaluate model on a single trajectory

        Args:
            h5_path: Path to h5 file containing [h, mx, my]
            output_dir: Directory to save results
            visualize: Whether to generate visualization

        Returns:
            Result dictionary with per-field metrics
        """
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]  # [T, 3, H, W]

        T, C, H, W = data.shape
        assert C == 3, f"Expected 3 channels (h, mx, my), got {C}"

        # Extract fields
        h_true = data[:, 0]   # [T, H, W]
        mx_true = data[:, 1]  # [T, H, W]
        my_true = data[:, 2]  # [T, H, W]

        # Rollout prediction
        num_steps = (T - 1) // self.ndt
        predictions = []
        current_state = torch.tensor(data[0:1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for step in range(num_steps):
                output = self.model(current_state)
                if isinstance(output, tuple):
                    output = output[0]
                predictions.append(output.cpu().numpy())
                current_state = output

        predictions = np.concatenate(predictions, axis=0)  # [num_steps, 3, H, W]

        # Extract predicted fields
        h_pred = predictions[:, 0]
        mx_pred = predictions[:, 1]
        my_pred = predictions[:, 2]

        # Compute per-field metrics
        results = {
            'trajectory': os.path.basename(h5_path),
            'num_steps': num_steps,
            'grid_size': [H, W],
        }

        # Get ground truth at prediction times
        pred_times = [(i + 1) * self.ndt for i in range(num_steps)]
        valid_times = [t for t in pred_times if t < T]
        num_valid = len(valid_times)

        if num_valid > 0:
            h_gt = np.stack([h_true[t] for t in valid_times])
            mx_gt = np.stack([mx_true[t] for t in valid_times])
            my_gt = np.stack([my_true[t] for t in valid_times])

            # Per-field MAE
            results['h_mae'] = float(np.abs(h_pred[:num_valid] - h_gt).mean())
            results['mx_mae'] = float(np.abs(mx_pred[:num_valid] - mx_gt).mean())
            results['my_mae'] = float(np.abs(my_pred[:num_valid] - my_gt).mean())

            # Per-field RMSE
            results['h_rmse'] = float(np.sqrt(((h_pred[:num_valid] - h_gt) ** 2).mean()))
            results['mx_rmse'] = float(np.sqrt(((mx_pred[:num_valid] - mx_gt) ** 2).mean()))
            results['my_rmse'] = float(np.sqrt(((my_pred[:num_valid] - my_gt) ** 2).mean()))

            # Overall MAE (all fields)
            results['overall_mae'] = float(np.abs(predictions[:num_valid] - data[valid_times]).mean())

            # Error at final time
            results['h_mae_final'] = float(np.abs(h_pred[num_valid-1] - h_gt[-1]).mean())
            results['mx_mae_final'] = float(np.abs(mx_pred[num_valid-1] - mx_gt[-1]).mean())
            results['my_mae_final'] = float(np.abs(my_pred[num_valid-1] - my_gt[-1]).mean())

        # Water depth violation (h < 0)
        h_violations = h_pred < 0
        results['h_violation_rate'] = float(h_violations.mean() * 100)  # percentage
        results['h_min'] = float(h_pred.min())
        results['h_max'] = float(h_pred.max())

        # Conditional mean violation magnitude
        if h_violations.any():
            results['h_violation_magnitude'] = float(np.abs(h_pred[h_violations]).mean())
        else:
            results['h_violation_magnitude'] = 0.0

        # Mass conservation for h
        initial_mass = h_true[0].sum()
        pred_masses = [h_pred[i].sum() for i in range(num_steps)]
        mass_drifts = [abs(m - initial_mass) / (abs(initial_mass) + 1e-8) for m in pred_masses]
        results['h_mass_drift_mean'] = float(np.mean(mass_drifts))
        results['h_mass_drift_max'] = float(np.max(mass_drifts))
        results['h_mass_drift_final'] = float(mass_drifts[-1]) if mass_drifts else 0.0

        # Classify case based on initial condition
        results['case_type'] = self._classify_case(h_true[0], mx_true[0], my_true[0])

        # Visualization
        if visualize and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            traj_name = os.path.splitext(os.path.basename(h5_path))[0]
            self._visualize_trajectory(
                h_true, mx_true, my_true,
                h_pred, mx_pred, my_pred,
                pred_times[:num_valid],
                os.path.join(output_dir, f'{traj_name}_comparison.png')
            )
            self._plot_error_evolution(
                h_true, mx_true, my_true,
                h_pred, mx_pred, my_pred,
                pred_times[:num_valid],
                os.path.join(output_dir, f'{traj_name}_errors.png')
            )

        return results

    def _classify_case(self, h0: np.ndarray, mx0: np.ndarray, my0: np.ndarray) -> str:
        """
        Classify initial condition case type

        Types:
        - A1: Dam break (sharp gradient in h)
        - A2: Gaussian bump
        - B1: Uniform h with momentum
        - B2: Complex initial condition
        """
        h_range = h0.max() - h0.min()
        h_gradient = np.abs(np.gradient(h0)).max()
        has_momentum = np.abs(mx0).max() > 1e-6 or np.abs(my0).max() > 1e-6

        # Heuristic classification
        if h_gradient > 0.5 * h_range:
            return "A1_dam_break"
        elif h_range > 0.1 and not has_momentum:
            return "A2_gaussian"
        elif h_range < 0.1 and has_momentum:
            return "B1_uniform_momentum"
        else:
            return "B2_complex"

    def _visualize_trajectory(
        self,
        h_true: np.ndarray, mx_true: np.ndarray, my_true: np.ndarray,
        h_pred: np.ndarray, mx_pred: np.ndarray, my_pred: np.ndarray,
        times: List[int],
        save_path: str
    ):
        """
        Visualize trajectory comparison

        Shows ground truth and prediction for h field at selected times
        """
        num_times = min(5, len(times))
        time_indices = np.linspace(0, len(times) - 1, num_times, dtype=int)

        fig, axes = plt.subplots(3, num_times, figsize=(4 * num_times, 10))

        for col, idx in enumerate(time_indices):
            t = times[idx]

            # Ground truth h
            if t < len(h_true):
                im0 = axes[0, col].imshow(h_true[t], cmap='Blues', vmin=0)
                axes[0, col].set_title(f't={t} (GT)')
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

            # Predicted h
            im1 = axes[1, col].imshow(h_pred[idx], cmap='Blues', vmin=0)
            axes[1, col].set_title(f't={t} (Pred)')
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

            # Error
            if t < len(h_true):
                error = np.abs(h_pred[idx] - h_true[t])
                im2 = axes[2, col].imshow(error, cmap='Reds')
                axes[2, col].set_title(f'|Error| (MAE={error.mean():.4f})')
                plt.colorbar(im2, ax=axes[2, col], fraction=0.046)

        axes[0, 0].set_ylabel('Ground Truth h')
        axes[1, 0].set_ylabel('Prediction h')
        axes[2, 0].set_ylabel('Absolute Error')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_error_evolution(
        self,
        h_true: np.ndarray, mx_true: np.ndarray, my_true: np.ndarray,
        h_pred: np.ndarray, mx_pred: np.ndarray, my_pred: np.ndarray,
        times: List[int],
        save_path: str
    ):
        """
        Plot error evolution over time for each field
        """
        num_valid = len(times)
        h_errors = []
        mx_errors = []
        my_errors = []
        mass_drifts = []

        initial_mass = h_true[0].sum()

        for i, t in enumerate(times):
            if t < len(h_true):
                h_errors.append(np.abs(h_pred[i] - h_true[t]).mean())
                mx_errors.append(np.abs(mx_pred[i] - mx_true[t]).mean())
                my_errors.append(np.abs(my_pred[i] - my_true[t]).mean())

            pred_mass = h_pred[i].sum()
            mass_drifts.append(abs(pred_mass - initial_mass) / (abs(initial_mass) + 1e-8))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Per-field MAE
        axes[0].plot(times[:len(h_errors)], h_errors, 'b-', label='h', linewidth=2)
        axes[0].plot(times[:len(mx_errors)], mx_errors, 'g-', label='mx', linewidth=2)
        axes[0].plot(times[:len(my_errors)], my_errors, 'r-', label='my', linewidth=2)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Per-Field Error Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mass conservation
        axes[1].plot(times[:len(mass_drifts)], np.array(mass_drifts) * 100, 'k-', linewidth=2)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Relative Mass Drift (%)')
        axes[1].set_title('Mass Conservation (h field)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_shallow_water(
    model,
    test_folder: str,
    output_dir: str,
    ndt: int = 1,
    device=None,
    visualize: bool = True,
    max_trajectories: Optional[int] = None
) -> Dict:
    """
    Evaluate shallow water model on test set

    Args:
        model: PyTorch model
        test_folder: Folder containing test h5 files
        output_dir: Directory to save results
        ndt: Time step interval
        device: Computation device
        visualize: Whether to generate visualizations
        max_trajectories: Maximum number of trajectories to evaluate

    Returns:
        Summary dictionary with per-field and per-case statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = ShallowWaterEvaluator(model, device, ndt)

    # Find all h5 files
    h5_files = sorted([
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if f.endswith('.h5')
    ])

    if max_trajectories is not None:
        h5_files = h5_files[:max_trajectories]

    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each trajectory
    all_results = []
    case_results = {}

    for h5_path in tqdm(h5_files, desc='Evaluating shallow water'):
        result = evaluator.evaluate_trajectory(
            h5_path,
            output_dir=os.path.join(output_dir, 'trajectories') if visualize else None,
            visualize=visualize
        )
        all_results.append(result)

        # Group by case type
        case_type = result['case_type']
        if case_type not in case_results:
            case_results[case_type] = []
        case_results[case_type].append(result)

    # Compute summary statistics
    summary = {
        'num_trajectories': len(all_results),
        'ndt': ndt,
    }

    # Overall metrics (time-averaged)
    for key in ['h_mae', 'mx_mae', 'my_mae', 'h_rmse', 'mx_rmse', 'my_rmse',
                'overall_mae', 'h_violation_rate', 'h_violation_magnitude',
                'h_mass_drift_mean', 'h_mass_drift_max', 'h_mass_drift_final',
                'h_mae_final', 'mx_mae_final', 'my_mae_final']:
        values = [r[key] for r in all_results if key in r]
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))

    # @T time point statistics for rollout (T = 0.25, 0.5, 0.75, 1.0)
    # Using final step results as proxy for T=1.0
    summary['mae_at_T1.0_h'] = summary.get('h_mae_final_mean', 0)
    summary['mae_at_T1.0_mx'] = summary.get('mx_mae_final_mean', 0)
    summary['mae_at_T1.0_my'] = summary.get('my_mae_final_mean', 0)

    # Per-case statistics (for shallow water, cases are A1, A2, B1, B2)
    summary['per_case'] = {}
    for case_type, results in case_results.items():
        case_summary = {'count': len(results)}
        # Per-field MAE
        for key in ['h_mae', 'mx_mae', 'my_mae', 'overall_mae',
                    'h_violation_rate', 'h_violation_magnitude',
                    'h_mass_drift_mean', 'h_mass_drift_final']:
            values = [r[key] for r in results if key in r]
            if values:
                case_summary[f'{key}_mean'] = float(np.mean(values))
                case_summary[f'{key}_std'] = float(np.std(values))
        summary['per_case'][case_type] = case_summary

    # Save summary
    with open(os.path.join(output_dir, 'sw_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    with open(os.path.join(output_dir, 'sw_detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate summary plots
    _plot_summary(all_results, output_dir)

    print(f"\nShallow Water Evaluation Summary:")
    print(f"  Trajectories: {summary['num_trajectories']}")
    print(f"  h MAE: {summary.get('h_mae_mean', 'N/A'):.4e} +/- {summary.get('h_mae_std', 0):.4e}")
    print(f"  mx MAE: {summary.get('mx_mae_mean', 'N/A'):.4e}")
    print(f"  my MAE: {summary.get('my_mae_mean', 'N/A'):.4e}")
    print(f"  h Violation Rate: {summary.get('h_violation_rate_mean', 'N/A'):.2f}%")
    print(f"  h Mass Drift: {summary.get('h_mass_drift_mean_mean', 'N/A'):.4e}")

    return summary


def _plot_summary(results: List[Dict], output_dir: str):
    """Generate summary plots with error bars for three fields"""

    # 1. Per-field error distribution (histogram)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, field in zip(axes, ['h', 'mx', 'my']):
        values = [r[f'{field}_mae'] for r in results if f'{field}_mae' in r]
        if values:
            ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='r', linestyle='--',
                       label=f'Mean: {mean_val:.4e}±{std_val:.4e}')
            ax.set_xlabel(f'{field} MAE')
            ax.set_ylabel('Count')
            ax.set_title(f'{field} Field Error Distribution')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'field_error_distribution.png'), dpi=150)
    plt.close()

    # 2. Per-field error bar chart (mean ± std)
    fig, ax = plt.subplots(figsize=(10, 6))

    field_names = ['h', 'mx', 'my']
    field_labels = ['Water Depth (h)', 'x-Momentum (mx)', 'y-Momentum (my)']
    means = []
    stds = []

    for field in field_names:
        values = [r[f'{field}_mae'] for r in results if f'{field}_mae' in r]
        if values:
            means.append(np.mean(values))
            stds.append(np.std(values))
        else:
            means.append(0)
            stds.append(0)

    x_pos = np.arange(len(field_names))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(field_labels)
    ax.set_ylabel('MAE')
    ax.set_title('Per-Field MAE (mean ± std)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'field_error_bars.png'), dpi=150)
    plt.close()

    # 3. Conservation drift per field
    fig, ax = plt.subplots(figsize=(10, 6))

    drift_fields = ['h_mass_drift_mean']
    drift_labels = ['h Mass Drift']
    drift_means = []
    drift_stds = []

    for key in drift_fields:
        values = [r[key] for r in results if key in r]
        if values:
            drift_means.append(np.mean(values))
            drift_stds.append(np.std(values))

    if drift_means:
        x_pos = np.arange(len(drift_fields))
        ax.bar(x_pos, drift_means, yerr=drift_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(drift_labels)
        ax.set_ylabel('Relative Drift')
        ax.set_title('Conservation Drift (mean ± std)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conservation_drift_bars.png'), dpi=150)
    plt.close()

    # 4. Violation rate and conditional magnitude
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violation rate histogram
    viol_rates = [r['h_violation_rate'] for r in results if 'h_violation_rate' in r]
    if viol_rates:
        axes[0].hist(viol_rates, bins=20, edgecolor='black', alpha=0.7)
        mean_rate = np.mean(viol_rates)
        std_rate = np.std(viol_rates)
        axes[0].axvline(mean_rate, color='r', linestyle='--',
                        label=f'Mean: {mean_rate:.2f}%±{std_rate:.2f}%')
        axes[0].set_xlabel('h Violation Rate (%)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Water Depth Violation Rate Distribution')
        axes[0].legend()

    # Conditional violation magnitude histogram
    viol_mags = [r['h_violation_magnitude'] for r in results
                 if 'h_violation_magnitude' in r and r['h_violation_magnitude'] > 0]
    if viol_mags:
        axes[1].hist(viol_mags, bins=20, edgecolor='black', alpha=0.7, color='orange')
        mean_mag = np.mean(viol_mags)
        axes[1].axvline(mean_mag, color='r', linestyle='--',
                        label=f'Mean: {mean_mag:.4e}')
        axes[1].set_xlabel('Conditional Violation Magnitude')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Conditional Mean OOB Magnitude (h < 0)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No violations', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Conditional Mean OOB Magnitude')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violation_distribution.png'), dpi=150)
    plt.close()

    # 5. Per-case comparison (if multiple cases exist)
    case_types = list(set(r.get('case_type', 'unknown') for r in results))
    if len(case_types) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))

        case_data = {}
        for case_type in case_types:
            case_results_list = [r for r in results if r.get('case_type') == case_type]
            h_maes = [r['h_mae'] for r in case_results_list if 'h_mae' in r]
            if h_maes:
                case_data[case_type] = (np.mean(h_maes), np.std(h_maes), len(h_maes))

        if case_data:
            sorted_cases = sorted(case_data.keys())
            x_pos = np.arange(len(sorted_cases))
            means = [case_data[c][0] for c in sorted_cases]
            stds = [case_data[c][1] for c in sorted_cases]
            counts = [case_data[c][2] for c in sorted_cases]

            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{c}\n(n={counts[i]})' for i, c in enumerate(sorted_cases)])
            ax.set_ylabel('h MAE')
            ax.set_title('Per-Case h MAE Comparison (mean ± std)')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_case_comparison.png'), dpi=150)
        plt.close()


if __name__ == "__main__":
    print("Shallow Water Evaluator")
    print("Usage: from src.evaluation.evaluator_sw import evaluate_shallow_water")
