"""
统一评估器 - 支持全测试集评估和详细指标统计

功能:
1. 单轨迹评估 (onestep + rollout) - 带完整可视化
2. 批量化测试集评估 (高效)
3. 守恒误差统计 (自适应处理初始值为0的情况)
4. 越界率/越界幅度统计
5. Rollout 误差 vs Horizon
6. 完整可视化输出
7. .dat文件导出
"""

import os
import h5py
import torch
import numpy as np
import joblib
import matplotlib

matplotlib.use('Agg')  # 强制使用无头模式，只生成文件不弹窗
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """单个轨迹的评估指标"""
    # 基本误差
    mae: List[float] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)

    # 守恒误差
    conservation_drift: List[float] = field(default_factory=list)  # 自适应漂移(相对或绝对)
    conservation_absolute: List[float] = field(default_factory=list)  # 绝对漂移
    total_mass: List[float] = field(default_factory=list)  # 预测总量
    true_mass: List[float] = field(default_factory=list)  # 真实总量

    # 越界统计
    violation_rate_lower: List[float] = field(default_factory=list)
    violation_rate_upper: List[float] = field(default_factory=list)
    violation_magnitude_lower: List[float] = field(default_factory=list)
    violation_magnitude_upper: List[float] = field(default_factory=list)

    # 条件平均越界幅度 (Conditional Mean OOB Magnitude) - 只统计越界点
    cond_magnitude_lower: List[float] = field(default_factory=list)
    cond_magnitude_upper: List[float] = field(default_factory=list)

    # 极值（预测）
    min_values: List[float] = field(default_factory=list)
    max_values: List[float] = field(default_factory=list)

    # 极值（真实场）
    true_min_values: List[float] = field(default_factory=list)
    true_max_values: List[float] = field(default_factory=list)

    # 浅水方程三场误差
    mae_h: List[float] = field(default_factory=list)
    mae_mx: List[float] = field(default_factory=list)
    mae_my: List[float] = field(default_factory=list)

    # 浅水方程守恒漂移 - h场（通常不为0，用相对误差）
    cons_drift_h: List[float] = field(default_factory=list)

    # 浅水方程守恒漂移 - mx场（分开统计）
    cons_drift_mx_rel: List[float] = field(default_factory=list)  # 初始值非0时的相对误差
    cons_drift_mx_abs: List[float] = field(default_factory=list)  # 初始值~0时的绝对误差

    # 浅水方程守恒漂移 - my场（分开统计）
    cons_drift_my_rel: List[float] = field(default_factory=list)  # 初始值非0时的相对误差
    cons_drift_my_abs: List[float] = field(default_factory=list)  # 初始值~0时的绝对误差

    # 浅水方程h场越界统计
    h_violation_rate_lower: List[float] = field(default_factory=list)
    h_cond_magnitude_lower: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def compute_conservation_error(pred_sum: float, initial_sum: float,
                               zero_threshold: float = 1e-12) -> Dict[str, float]:
    """
    自适应计算守恒误差

    对于初始值接近0的情况（如初始动量为0），使用绝对误差而非相对误差

    Args:
        pred_sum: 预测场的总和
        initial_sum: 初始场的总和
        zero_threshold: 判断初始值是否为0的阈值

    Returns:
        dict: {
            'absolute_drift': 绝对漂移,
            'relative_drift': 相对漂移（初始值~0时为None）,
            'is_zero_initial': 初始值是否接近0,
            'drift': 推荐使用的漂移值（自适应选择）
        }
    """
    absolute_drift = abs(pred_sum - initial_sum)

    if abs(initial_sum) > zero_threshold:
        # 初始值足够大，使用相对误差
        relative_drift = absolute_drift / abs(initial_sum)
        return {
            'absolute_drift': absolute_drift,
            'relative_drift': relative_drift,
            'is_zero_initial': False,
            'drift': relative_drift
        }
    else:
        # 初始值接近0，使用绝对误差
        return {
            'absolute_drift': absolute_drift,
            'relative_drift': None,
            'is_zero_initial': True,
            'drift': absolute_drift
        }


def compute_bound_violation(field: np.ndarray, lower_bound: Optional[float] = None,
                            upper_bound: Optional[float] = None, eps: float = 1e-6) -> Dict[str, float]:
    """
    计算越界统计

    返回:
        rate_lower/upper: 越界率 (%)
        magnitude_lower/upper: 全场平均越界幅度 (包含未越界点，值为0)
        cond_magnitude_lower/upper: 条件平均越界幅度 (Conditional Mean OOB Magnitude, 只统计越界点)
        min_value, max_value: 场的极值
    """
    total_points = field.size
    stats = {
        'rate_lower': 0.0,
        'rate_upper': 0.0,
        'magnitude_lower': 0.0,
        'magnitude_upper': 0.0,
        'cond_magnitude_lower': 0.0,
        'cond_magnitude_upper': 0.0,
        'min_value': float(field.min()),
        'max_value': float(field.max()),
    }

    if lower_bound is not None:
        violations_lower = field < (lower_bound - eps)
        num_violations = np.sum(violations_lower)
        stats['rate_lower'] = num_violations / total_points * 100
        if np.any(violations_lower):
            stats['magnitude_lower'] = np.mean(np.maximum(0, lower_bound - field))
            stats['cond_magnitude_lower'] = np.mean(lower_bound - field[violations_lower])

    if upper_bound is not None:
        violations_upper = field > (upper_bound + eps)
        num_violations = np.sum(violations_upper)
        stats['rate_upper'] = num_violations / total_points * 100
        if np.any(violations_upper):
            stats['magnitude_upper'] = np.mean(np.maximum(0, field - upper_bound))
            stats['cond_magnitude_upper'] = np.mean(field[violations_upper] - upper_bound)

    return stats


class UnifiedEvaluator:
    """统一评估器"""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            dataset_type: str,
            ndt: int = 1,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            zero_threshold: float = 1e-12  # 判断初始值是否为0的阈值
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataset_type = dataset_type
        self.ndt = ndt
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.zero_threshold = zero_threshold

        # 获取模型输出格式
        model_name = model.__class__.__name__
        if 'FluxNet_D' in model_name or 'FluxGNN_D' in model_name or 'FluxNet_D_Dirichlet' in model_name:
            self.output_format = 'fluxnet_d'
        elif 'FluxNet_SW_2D' == model_name:
            self.output_format = 'fluxnet_sw'
        elif 'FluxNet_SW_Baseline' == model_name:
            self.output_format = 'sw_baseline'
        elif 'FluxNet' in model_name:
            self.output_format = 'fluxnet_ul'
        else:
            self.output_format = 'cnn'

    def _load_h5_data(self, h5_file_path: str) -> Dict:
        """根据数据集类型加载h5数据，使用ndt进行采样"""
        with h5py.File(h5_file_path, 'r') as f:
            if self.dataset_type == 'convection_diffusion':
                return {
                    'c': f['c'][::self.ndt].astype(np.float32),
                    'u': f['u'][:].astype(np.float32)
                }
            elif self.dataset_type == 'traffic_flow':
                return {
                    'rho': f['rho'][::self.ndt].astype(np.float32),
                    'vmax': f['vmax'][:].astype(np.float32)
                }
            elif self.dataset_type == 'shallow_water':
                return {
                    'h': f['h'][::self.ndt].astype(np.float32),
                    'mx': f['mx'][::self.ndt].astype(np.float32),
                    'my': f['my'][::self.ndt].astype(np.float32)
                }
            elif self.dataset_type == 'spinodal_decomposition':
                return {
                    'phi': f['phi_data'][::self.ndt].astype(np.float32)
                }
            else:
                raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def _prepare_input(self, data: Dict, t: int) -> torch.Tensor:
        """准备模型输入"""
        if self.dataset_type == 'convection_diffusion':
            input_array = np.stack([data['c'][t], data['u']], axis=0)
        elif self.dataset_type == 'traffic_flow':
            input_array = np.stack([data['rho'][t], data['vmax']], axis=0)
        elif self.dataset_type == 'shallow_water':
            input_array = np.stack([data['h'][t], data['mx'][t], data['my'][t]], axis=0)
        elif self.dataset_type == 'spinodal_decomposition':
            input_array = data['phi'][t][np.newaxis, :]
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return torch.from_numpy(input_array).unsqueeze(0).to(self.device)

    def _get_conserved_field(self, data: Dict, t: int) -> np.ndarray:
        """获取守恒场"""
        if self.dataset_type == 'convection_diffusion':
            return data['c'][t]
        elif self.dataset_type == 'traffic_flow':
            return data['rho'][t]
        elif self.dataset_type == 'shallow_water':
            return np.stack([data['h'][t], data['mx'][t], data['my'][t]], axis=0)
        elif self.dataset_type == 'spinodal_decomposition':
            return data['phi'][t]
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def _get_prediction(self, model_output) -> np.ndarray:
        """从模型输出中获取预测"""
        if self.output_format == 'fluxnet_d':
            pred = model_output[0]
        elif self.output_format in ['fluxnet_sw', 'sw_baseline']:
            pred = model_output[0]
        elif self.output_format == 'fluxnet_ul':
            pred = model_output[0]
        else:
            pred = model_output[0]

        return pred.squeeze().cpu().numpy()

    def _compute_shallow_water_conservation(self, pred: np.ndarray, initial_field: np.ndarray,
                                            metrics: EvaluationMetrics):
        """
        计算浅水方程三场的守恒误差（自适应处理初始值为0的情况）
        """
        # h场 - 质量通常不为0，直接用相对误差
        cons_result_h = compute_conservation_error(
            pred[0].sum(), initial_field[0].sum(), self.zero_threshold
        )
        metrics.cons_drift_h.append(cons_result_h['drift'])

        # mx场 - 根据初始值情况分类统计
        cons_result_mx = compute_conservation_error(
            pred[1].sum(), initial_field[1].sum(), self.zero_threshold
        )
        if cons_result_mx['is_zero_initial']:
            metrics.cons_drift_mx_abs.append(cons_result_mx['absolute_drift'])
        else:
            metrics.cons_drift_mx_rel.append(cons_result_mx['relative_drift'])

        # my场 - 根据初始值情况分类统计
        cons_result_my = compute_conservation_error(
            pred[2].sum(), initial_field[2].sum(), self.zero_threshold
        )
        if cons_result_my['is_zero_initial']:
            metrics.cons_drift_my_abs.append(cons_result_my['absolute_drift'])
        else:
            metrics.cons_drift_my_rel.append(cons_result_my['relative_drift'])

    def evaluate_single_trajectory_onestep(
            self,
            h5_file_path: str,
            save_dir: Optional[str] = None,
            visualize: bool = True
    ) -> EvaluationMetrics:
        """
        Onestep评估单个轨迹

        每一步都用真实值作为输入进行预测
        """
        data = self._load_h5_data(h5_file_path)
        metrics = EvaluationMetrics()

        # 获取时间步数
        if self.dataset_type == 'shallow_water':
            num_steps = data['h'].shape[0] - 1
        elif self.dataset_type == 'convection_diffusion':
            num_steps = data['c'].shape[0] - 1
        elif self.dataset_type == 'traffic_flow':
            num_steps = data['rho'].shape[0] - 1
        else:
            num_steps = data['phi'].shape[0] - 1

        # 存储所有预测和真实值用于可视化
        all_preds = []
        all_targets = []

        # 初始总量
        initial_field = self._get_conserved_field(data, 0)
        if len(initial_field.shape) > 2:  # 多通道
            initial_mass = initial_field[0].sum()
        else:
            initial_mass = initial_field.sum()

        # 逐步评估
        for t in range(num_steps):
            input_tensor = self._prepare_input(data, t)
            target = self._get_conserved_field(data, t + 1)

            with torch.no_grad():
                output = self.model(input_tensor)
                pred = self._get_prediction(output)

            all_preds.append(pred.copy())
            all_targets.append(target.copy())

            # 计算误差
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            metrics.mae.append(mae)
            metrics.rmse.append(rmse)

            # 守恒性（主场）
            if len(pred.shape) > 2:
                pred_mass = pred[0].sum()
                true_mass = target[0].sum()
            else:
                pred_mass = pred.sum()
                true_mass = target.sum()

            cons_result = compute_conservation_error(pred_mass, initial_mass, self.zero_threshold)
            metrics.conservation_drift.append(cons_result['drift'])
            metrics.conservation_absolute.append(cons_result['absolute_drift'])
            metrics.total_mass.append(pred_mass)
            metrics.true_mass.append(true_mass)

            # 越界统计
            if len(pred.shape) > 2:
                pred_field = pred[0]
                target_field = target[0]
            else:
                pred_field = pred
                target_field = target

            viol_stats = compute_bound_violation(pred_field, self.lower_bound, self.upper_bound)
            metrics.violation_rate_lower.append(viol_stats['rate_lower'])
            metrics.violation_rate_upper.append(viol_stats['rate_upper'])
            metrics.violation_magnitude_lower.append(viol_stats['magnitude_lower'])
            metrics.violation_magnitude_upper.append(viol_stats['magnitude_upper'])
            metrics.cond_magnitude_lower.append(viol_stats['cond_magnitude_lower'])
            metrics.cond_magnitude_upper.append(viol_stats['cond_magnitude_upper'])
            metrics.min_values.append(viol_stats['min_value'])
            metrics.max_values.append(viol_stats['max_value'])

            # 真实场极值
            metrics.true_min_values.append(float(target_field.min()))
            metrics.true_max_values.append(float(target_field.max()))

            # 浅水方程三场分别统计
            if self.dataset_type == 'shallow_water':
                for i, (name, field_pred, field_true) in enumerate([
                    ('h', pred[0], target[0]),
                    ('mx', pred[1], target[1]),
                    ('my', pred[2], target[2])
                ]):
                    mae_i = np.mean(np.abs(field_pred - field_true))
                    if name == 'h':
                        metrics.mae_h.append(mae_i)
                        # h场越界统计 (仅下界)
                        h_viol = compute_bound_violation(field_pred, lower_bound=0.0, upper_bound=None)
                        metrics.h_violation_rate_lower.append(h_viol['rate_lower'])
                        metrics.h_cond_magnitude_lower.append(h_viol['cond_magnitude_lower'])
                    elif name == 'mx':
                        metrics.mae_mx.append(mae_i)
                    elif name == 'my':
                        metrics.mae_my.append(mae_i)

                # 统一处理浅水方程守恒误差
                self._compute_shallow_water_conservation(pred, initial_field, metrics)

        # 可视化
        if visualize and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._visualize_evaluation(
                all_preds, all_targets, metrics, save_dir, 'onestep',
                h5_file_path
            )

        return metrics

    def evaluate_single_trajectory_rollout(
            self,
            h5_file_path: str,
            save_dir: Optional[str] = None,
            visualize: bool = True
    ) -> EvaluationMetrics:
        """
        Rollout评估单个轨迹

        自回归预测：使用预测值作为下一步的输入
        """
        data = self._load_h5_data(h5_file_path)
        metrics = EvaluationMetrics()

        # 获取时间步数
        if self.dataset_type == 'shallow_water':
            num_steps = data['h'].shape[0] - 1
        elif self.dataset_type == 'convection_diffusion':
            num_steps = data['c'].shape[0] - 1
        elif self.dataset_type == 'traffic_flow':
            num_steps = data['rho'].shape[0] - 1
        else:
            num_steps = data['phi'].shape[0] - 1

        all_preds = []
        all_targets = []

        # 初始状态
        initial_field = self._get_conserved_field(data, 0)
        if len(initial_field.shape) > 2:
            initial_mass = initial_field[0].sum()
        else:
            initial_mass = initial_field.sum()

        # 当前预测状态
        current_pred = initial_field.copy()

        for t in range(num_steps):
            # 准备输入 (使用预测值)
            if self.dataset_type == 'convection_diffusion':
                input_array = np.stack([current_pred, data['u']], axis=0)
            elif self.dataset_type == 'traffic_flow':
                input_array = np.stack([current_pred, data['vmax']], axis=0)
            elif self.dataset_type == 'shallow_water':
                input_array = current_pred
            else:  # spinodal
                input_array = current_pred[np.newaxis, :] if len(current_pred.shape) == 2 else current_pred

            input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().to(self.device)

            target = self._get_conserved_field(data, t + 1)

            with torch.no_grad():
                output = self.model(input_tensor)
                pred = self._get_prediction(output)

            all_preds.append(pred.copy())
            all_targets.append(target.copy())

            # 更新当前预测
            current_pred = pred.copy()

            # 计算误差
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            metrics.mae.append(mae)
            metrics.rmse.append(rmse)

            # 守恒性
            if len(pred.shape) > 2:
                pred_mass = pred[0].sum()
                true_mass = target[0].sum()
            else:
                pred_mass = pred.sum()
                true_mass = target.sum()

            cons_result = compute_conservation_error(pred_mass, initial_mass, self.zero_threshold)
            metrics.conservation_drift.append(cons_result['drift'])
            metrics.conservation_absolute.append(cons_result['absolute_drift'])
            metrics.total_mass.append(pred_mass)
            metrics.true_mass.append(true_mass)

            # 越界统计
            if len(pred.shape) > 2:
                pred_field = pred[0]
                target_field = target[0]
            else:
                pred_field = pred
                target_field = target

            viol_stats = compute_bound_violation(pred_field, self.lower_bound, self.upper_bound)
            metrics.violation_rate_lower.append(viol_stats['rate_lower'])
            metrics.violation_rate_upper.append(viol_stats['rate_upper'])
            metrics.violation_magnitude_lower.append(viol_stats['magnitude_lower'])
            metrics.violation_magnitude_upper.append(viol_stats['magnitude_upper'])
            metrics.cond_magnitude_lower.append(viol_stats['cond_magnitude_lower'])
            metrics.cond_magnitude_upper.append(viol_stats['cond_magnitude_upper'])
            metrics.min_values.append(viol_stats['min_value'])
            metrics.max_values.append(viol_stats['max_value'])

            # 真实场极值
            metrics.true_min_values.append(float(target_field.min()))
            metrics.true_max_values.append(float(target_field.max()))

            # 浅水方程三场分别统计
            if self.dataset_type == 'shallow_water':
                for i, (name, field_pred, field_true) in enumerate([
                    ('h', pred[0], target[0]),
                    ('mx', pred[1], target[1]),
                    ('my', pred[2], target[2])
                ]):
                    mae_i = np.mean(np.abs(field_pred - field_true))
                    if name == 'h':
                        metrics.mae_h.append(mae_i)
                        # h场越界统计 (仅下界)
                        h_viol = compute_bound_violation(field_pred, lower_bound=0.0, upper_bound=None)
                        metrics.h_violation_rate_lower.append(h_viol['rate_lower'])
                        metrics.h_cond_magnitude_lower.append(h_viol['cond_magnitude_lower'])
                    elif name == 'mx':
                        metrics.mae_mx.append(mae_i)
                    elif name == 'my':
                        metrics.mae_my.append(mae_i)

                # 统一处理浅水方程守恒误差
                self._compute_shallow_water_conservation(pred, initial_field, metrics)

        # 可视化
        if visualize and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._visualize_evaluation(
                all_preds, all_targets, metrics, save_dir, 'rollout',
                h5_file_path
            )

        return metrics

    def _visualize_evaluation(
            self,
            all_preds: List[np.ndarray],
            all_targets: List[np.ndarray],
            metrics: EvaluationMetrics,
            save_dir: str,
            mode: str,
            h5_file_path: str
    ):
        """完整可视化"""
        os.makedirs(save_dir, exist_ok=True)
        num_steps = len(all_preds)

        # 1. 误差曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # MAE/RMSE
        axes[0, 0].plot(metrics.mae, 'b-', linewidth=2, label='MAE')
        axes[0, 0].plot(metrics.rmse, 'r--', linewidth=2, label='RMSE')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].set_title(f'{mode.upper()} Error')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 守恒曲线
        axes[0, 1].plot(metrics.total_mass, 'r-', linewidth=2, label='Predicted')
        axes[0, 1].plot(metrics.true_mass, 'b--', linewidth=2, label='True')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Total Mass')
        axes[0, 1].set_title('Conservation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 守恒漂移
        axes[1, 0].plot(metrics.conservation_drift, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Drift (Adaptive)')
        axes[1, 0].set_title('Conservation Drift')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 越界率
        axes[1, 1].plot(metrics.violation_rate_lower, 'b-', linewidth=2, label='Lower')
        axes[1, 1].plot(metrics.violation_rate_upper, 'r-', linewidth=2, label='Upper')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Violation Rate (%)')
        axes[1, 1].set_title('Bound Violation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{mode.upper()} Evaluation - {Path(h5_file_path).stem}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_metrics.png'), dpi=150)
        plt.close()

        # 2. 极值曲线（包含真实场）
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics.min_values, 'b-', linewidth=2, label='Pred Min')
        ax.plot(metrics.max_values, 'r-', linewidth=2, label='Pred Max')
        if metrics.true_min_values:
            ax.plot(metrics.true_min_values, 'b--', linewidth=1.5, alpha=0.7, label='True Min')
            ax.plot(metrics.true_max_values, 'r--', linewidth=1.5, alpha=0.7, label='True Max')
        if self.lower_bound is not None:
            ax.axhline(y=self.lower_bound, color='b', linestyle=':', alpha=0.5,
                       label=f'Lower Bound ({self.lower_bound})')
        if self.upper_bound is not None:
            ax.axhline(y=self.upper_bound, color='r', linestyle=':', alpha=0.5,
                       label=f'Upper Bound ({self.upper_bound})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'{mode.upper()} Min/Max Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_minmax.png'), dpi=150)
        plt.close()

        # 3. 场可视化 (选取几个关键时刻)
        key_steps = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
        key_steps = [s for s in key_steps if s < num_steps]

        if self.dataset_type in ['spinodal_decomposition']:
            self._visualize_2d_snapshots(all_preds, all_targets, key_steps, save_dir, mode)
        elif self.dataset_type in ['convection_diffusion', 'traffic_flow']:
            self._visualize_1d_snapshots(all_preds, all_targets, key_steps, save_dir, mode)
        elif self.dataset_type == 'shallow_water':
            self._visualize_sw_snapshots(all_preds, all_targets, key_steps, save_dir, mode)

        # 4. 保存数据
        plot_data = {
            'metrics': metrics.to_dict(),
            'all_preds': all_preds,
            'all_targets': all_targets,
            'mode': mode,
            'h5_file': h5_file_path
        }
        joblib.dump(plot_data, os.path.join(save_dir, f'{mode}_data.pkl'))

        # 5. 保存.dat文件 (2D场)
        if self.dataset_type in ['spinodal_decomposition', 'shallow_water']:
            self._save_dat_files(all_preds, all_targets, key_steps, save_dir, mode)

    def _visualize_2d_snapshots(self, all_preds, all_targets, key_steps, save_dir, mode):
        """2D场快照可视化"""
        n = len(key_steps)
        fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))

        for i, t in enumerate(key_steps):
            pred = all_preds[t]
            target = all_targets[t]

            if len(pred.shape) > 2:
                pred = pred[0]
                target = target[0]

            error = np.abs(pred - target)
            vmin = min(pred.min(), target.min())
            vmax = max(pred.max(), target.max())

            im0 = axes[0, i].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f't={t} Pred')
            plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

            im1 = axes[1, i].imshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f't={t} True')
            plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

            im2 = axes[2, i].imshow(error, cmap='hot')
            axes[2, i].set_title(f't={t} Error')
            plt.colorbar(im2, ax=axes[2, i], shrink=0.8)

        plt.suptitle(f'{mode.upper()} Field Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_snapshots.png'), dpi=150)
        plt.close()

    def _visualize_1d_snapshots(self, all_preds, all_targets, key_steps, save_dir, mode):
        """1D场快照可视化"""
        n = len(key_steps)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

        # 预先计算所有数据的范围
        pred_values = []
        target_values = []
        all_errors = []

        for t in key_steps:
            pred = all_preds[t]
            target = all_targets[t]

            if len(pred.shape) > 1:
                pred = pred[0]
                target = target[0]

            pred_values.extend(pred.flatten())
            target_values.extend(target.flatten())
            all_errors.extend(np.abs(pred - target).flatten())

        all_values = pred_values + target_values
        vmin = np.min(all_values)
        vmax = np.max(all_values)
        v_range = vmax - vmin
        v_margin = v_range * 0.05 if v_range > 0 else 0.1
        ylim_row1 = [vmin - v_margin, vmax + v_margin]

        emin = np.min(all_errors)
        emax = np.max(all_errors)
        e_range = emax - emin
        e_margin = e_range * 0.05 if e_range > 0 else 0.1
        ylim_row2 = [emin - e_margin, emax + e_margin]

        for i, t in enumerate(key_steps):
            pred = all_preds[t]
            target = all_targets[t]
            if len(pred.shape) > 1:
                pred = pred[0]
                target = target[0]
            x = np.arange(len(pred))

            axes[0, i].plot(x, target, 'b-', linewidth=2, label='True')
            axes[0, i].plot(x, pred, 'r--', linewidth=2, label='Pred')
            axes[0, i].set_title(f't={t}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_ylim(ylim_row1)

            error = np.abs(pred - target)
            axes[1, i].plot(x, error, 'g-', linewidth=2)
            axes[1, i].set_title(f'Error (MAE={error.mean():.2e})')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_ylim(ylim_row2)

        plt.suptitle(f'{mode.upper()} Field Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_snapshots.png'), dpi=150)
        plt.close()

        # 时空图
        pred_stack = np.array([p[0] if len(p.shape) > 1 else p for p in all_preds])
        target_stack = np.array([t[0] if len(t.shape) > 1 else t for t in all_targets])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        vmin = min(pred_stack.min(), target_stack.min())
        vmax = max(pred_stack.max(), target_stack.max())

        im0 = axes[0].imshow(pred_stack, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Predicted')
        axes[0].set_xlabel('Space')
        axes[0].set_ylabel('Time')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(target_stack, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('True')
        axes[1].set_xlabel('Space')
        plt.colorbar(im1, ax=axes[1])

        error_stack = np.abs(pred_stack - target_stack)
        im2 = axes[2].imshow(error_stack, aspect='auto', cmap='hot')
        axes[2].set_title('Error')
        axes[2].set_xlabel('Space')
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f'{mode.upper()} Space-Time Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_spacetime.png'), dpi=150)
        plt.close()

    def _visualize_sw_snapshots(self, all_preds, all_targets, key_steps, save_dir, mode):
        """浅水方程快照可视化"""
        field_names = ['h', 'mx', 'my']

        for t in key_steps:
            pred = all_preds[t]
            target = all_targets[t]

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))

            for i, name in enumerate(field_names):
                pred_f = pred[i]
                target_f = target[i]
                error_f = np.abs(pred_f - target_f)

                vmin = min(pred_f.min(), target_f.min())
                vmax = max(pred_f.max(), target_f.max())

                im0 = axes[i, 0].imshow(pred_f, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 0].set_title(f'Pred {name}')
                plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

                im1 = axes[i, 1].imshow(target_f, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 1].set_title(f'True {name}')
                plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

                im2 = axes[i, 2].imshow(error_f, cmap='hot')
                axes[i, 2].set_title(f'Error {name}')
                plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

            plt.suptitle(f'{mode.upper()} t={t}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{mode}_sw_t{t:04d}.png'), dpi=150)
            plt.close()

    def _save_dat_files(self, all_preds, all_targets, key_steps, save_dir, mode):
        """保存.dat文件"""
        dat_dir = os.path.join(save_dir, 'dat_files')
        os.makedirs(dat_dir, exist_ok=True)

        for t in key_steps:
            pred = all_preds[t]
            target = all_targets[t]

            if len(pred.shape) > 2:
                for c in range(pred.shape[0]):
                    self._write_dat(pred[c], os.path.join(dat_dir, f'{mode}_pred_c{c}_t{t:04d}.dat'))
                    self._write_dat(target[c], os.path.join(dat_dir, f'{mode}_true_c{c}_t{t:04d}.dat'))
            elif len(pred.shape) == 2:
                self._write_dat(pred, os.path.join(dat_dir, f'{mode}_pred_t{t:04d}.dat'))
                self._write_dat(target, os.path.join(dat_dir, f'{mode}_true_t{t:04d}.dat'))

    def _write_dat(self, field: np.ndarray, filepath: str):
        """写入tecplot格式dat文件"""
        if len(field.shape) != 2:
            return

        I, J = field.shape
        with open(filepath, 'w') as f:
            f.write(f'VARIABLE="x","y","value"\n')
            f.write(f'ZONE t="field", I={I}, J={J}, F=POINT\n')
            for value in field.flatten():
                f.write(f"{value:.6f}\n")

    def evaluate_test_set_batch(
            self,
            test_folder: str,
            output_dir: str,
            batch_size: int = 32,
            mode: str = 'onestep',
            visualize_trajectories: Optional[List[str]] = None
    ) -> Dict:
        """
        批量化评估整个测试集
        """
        os.makedirs(output_dir, exist_ok=True)

        h5_files = sorted(list(Path(test_folder).glob("*.h5")))
        print(f"\n找到 {len(h5_files)} 个测试轨迹")

        all_metrics = []

        if visualize_trajectories is None:
            # vis_files = h5_files[:3] if len(h5_files) >= 3 else h5_files
            vis_files = []
        elif visualize_trajectories == 'all':
            vis_files = h5_files
        else:
            vis_files = [Path(f) for f in visualize_trajectories if Path(f).exists()]

        for h5_file in tqdm(h5_files, desc=f"评估测试集 ({mode})"):
            need_vis = h5_file in vis_files

            if mode == 'onestep':
                metrics = self.evaluate_single_trajectory_onestep(
                    str(h5_file),
                    save_dir=os.path.join(output_dir, 'trajectories', h5_file.stem) if need_vis else None,
                    visualize=need_vis
                )
            else:
                metrics = self.evaluate_single_trajectory_rollout(
                    str(h5_file),
                    save_dir=os.path.join(output_dir, 'trajectories', h5_file.stem) if need_vis else None,
                    visualize=need_vis
                )

            all_metrics.append(metrics)

        # 汇总统计
        summary = self._aggregate_metrics(all_metrics)

        # 保存汇总结果
        joblib.dump(summary, os.path.join(output_dir, f'test_set_summary_{mode}.pkl'))

        import json
        json_summary = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in summary.items() if not isinstance(v, list)}
        with open(os.path.join(output_dir, f'test_set_summary_{mode}.json'), 'w') as f:
            json.dump(json_summary, f, indent=2)

        self._visualize_summary(all_metrics, summary, output_dir, mode)
        self._create_summary_gallery(h5_files, vis_files, output_dir, mode)
        self._print_summary(summary)

        return summary

    def _create_summary_gallery(self, h5_files: List[Path], vis_files: List[Path],
                                output_dir: str, mode: str):
        """创建汇总可视化文件夹"""
        gallery_dir = os.path.join(output_dir, f'all_{mode}')
        os.makedirs(gallery_dir, exist_ok=True)

        for h5_file in vis_files:
            traj_dir = os.path.join(output_dir, 'trajectories', h5_file.stem)
            if not os.path.exists(traj_dir):
                continue

            for img_name in [f'{mode}_metrics.png', f'{mode}_snapshots.png',
                             f'{mode}_minmax.png', f'{mode}_spacetime.png']:
                src_path = os.path.join(traj_dir, img_name)
                if os.path.exists(src_path):
                    dst_name = f'{h5_file.stem}_{img_name}'
                    import shutil
                    shutil.copy2(src_path, os.path.join(gallery_dir, dst_name))

        if len(vis_files) <= 16 and len(vis_files) > 0:
            self._create_combined_gallery_image(vis_files, output_dir, gallery_dir, mode)

    def _aggregate_metrics(self, all_metrics: List[EvaluationMetrics]) -> Dict:
        """汇总多个轨迹的指标"""
        max_steps = max(len(m.mae) for m in all_metrics)

        mae_by_step = [[] for _ in range(max_steps)]
        rmse_by_step = [[] for _ in range(max_steps)]
        cons_drift_by_step = [[] for _ in range(max_steps)]
        viol_lower_by_step = [[] for _ in range(max_steps)]
        viol_upper_by_step = [[] for _ in range(max_steps)]
        cond_mag_lower_by_step = [[] for _ in range(max_steps)]
        cond_mag_upper_by_step = [[] for _ in range(max_steps)]

        for m in all_metrics:
            for t, val in enumerate(m.mae):
                mae_by_step[t].append(val)
            for t, val in enumerate(m.rmse):
                rmse_by_step[t].append(val)
            for t, val in enumerate(m.conservation_drift):
                cons_drift_by_step[t].append(val)
            for t, val in enumerate(m.violation_rate_lower):
                viol_lower_by_step[t].append(val)
            for t, val in enumerate(m.violation_rate_upper):
                viol_upper_by_step[t].append(val)
            for t, val in enumerate(m.cond_magnitude_lower):
                cond_mag_lower_by_step[t].append(val)
            for t, val in enumerate(m.cond_magnitude_upper):
                cond_mag_upper_by_step[t].append(val)

        summary = {
            'num_trajectories': len(all_metrics),
            'max_steps': max_steps,

            'mae_mean_by_step': [np.mean(s) if s else 0 for s in mae_by_step],
            'mae_std_by_step': [np.std(s) if s else 0 for s in mae_by_step],
            'rmse_mean_by_step': [np.mean(s) if s else 0 for s in rmse_by_step],
            'rmse_std_by_step': [np.std(s) if s else 0 for s in rmse_by_step],
            'cons_drift_mean_by_step': [np.mean(s) if s else 0 for s in cons_drift_by_step],
            'cons_drift_std_by_step': [np.std(s) if s else 0 for s in cons_drift_by_step],
            'viol_lower_mean_by_step': [np.mean(s) if s else 0 for s in viol_lower_by_step],
            'viol_upper_mean_by_step': [np.mean(s) if s else 0 for s in viol_upper_by_step],
            'cond_mag_lower_mean_by_step': [np.mean([v for v in s if v > 0]) if any(v > 0 for v in s) else 0
                                            for s in cond_mag_lower_by_step],
            'cond_mag_upper_mean_by_step': [np.mean([v for v in s if v > 0]) if any(v > 0 for v in s) else 0
                                            for s in cond_mag_upper_by_step],

            'mae_overall_mean': np.mean([np.mean(m.mae) for m in all_metrics]),
            'mae_overall_std': np.std([np.mean(m.mae) for m in all_metrics]),
            'rmse_overall_mean': np.mean([np.mean(m.rmse) for m in all_metrics]),
            'rmse_overall_std': np.std([np.mean(m.rmse) for m in all_metrics]),
            'cons_drift_max': max(max(m.conservation_drift) for m in all_metrics),
            'cons_drift_mean': np.mean([np.mean(m.conservation_drift) for m in all_metrics]),
            'cons_drift_std': np.std([np.mean(m.conservation_drift) for m in all_metrics]),
            'viol_lower_mean': np.mean([np.mean(m.violation_rate_lower) for m in all_metrics]),
            'viol_lower_std': np.std([np.mean(m.violation_rate_lower) for m in all_metrics]),
            'viol_upper_mean': np.mean([np.mean(m.violation_rate_upper) for m in all_metrics]),
            'viol_upper_std': np.std([np.mean(m.violation_rate_upper) for m in all_metrics]),
            'min_value_overall': min(min(m.min_values) for m in all_metrics),
            'max_value_overall': max(max(m.max_values) for m in all_metrics),
        }

        all_cond_lower = [v for m in all_metrics for v in m.cond_magnitude_lower if v > 0]
        all_cond_upper = [v for m in all_metrics for v in m.cond_magnitude_upper if v > 0]
        summary['cond_magnitude_lower_mean'] = np.mean(all_cond_lower) if all_cond_lower else 0.0
        summary['cond_magnitude_lower_std'] = np.std(all_cond_lower) if all_cond_lower else 0.0
        summary['cond_magnitude_upper_mean'] = np.mean(all_cond_upper) if all_cond_upper else 0.0
        summary['cond_magnitude_upper_std'] = np.std(all_cond_upper) if all_cond_upper else 0.0

        for T_ratio in [0.25, 0.5, 0.75, 1.0]:
            T_idx = int(T_ratio * max_steps) - 1
            T_idx = max(0, min(T_idx, max_steps - 1))
            if mae_by_step[T_idx]:
                summary[f'mae_at_T{T_ratio}'] = np.mean(mae_by_step[T_idx])
                summary[f'mae_at_T{T_ratio}_std'] = np.std(mae_by_step[T_idx])
                summary[f'rmse_at_T{T_ratio}'] = np.mean(rmse_by_step[T_idx])
                summary[f'rmse_at_T{T_ratio}_std'] = np.std(rmse_by_step[T_idx])
                summary[f'cons_drift_at_T{T_ratio}'] = np.mean(cons_drift_by_step[T_idx])
                summary[f'cons_drift_at_T{T_ratio}_std'] = np.std(cons_drift_by_step[T_idx])
                summary[f'viol_lower_at_T{T_ratio}'] = np.mean(viol_lower_by_step[T_idx])
                summary[f'viol_lower_at_T{T_ratio}_std'] = np.std(viol_lower_by_step[T_idx])
                summary[f'viol_upper_at_T{T_ratio}'] = np.mean(viol_upper_by_step[T_idx])
                summary[f'viol_upper_at_T{T_ratio}_std'] = np.std(viol_upper_by_step[T_idx])

        # 浅水方程三场分别统计
        if all_metrics[0].mae_h:
            summary['sw_mae_h_mean'] = np.mean([np.mean(m.mae_h) for m in all_metrics])
            summary['sw_mae_h_std'] = np.std([np.mean(m.mae_h) for m in all_metrics])
            summary['sw_mae_mx_mean'] = np.mean([np.mean(m.mae_mx) for m in all_metrics])
            summary['sw_mae_mx_std'] = np.std([np.mean(m.mae_mx) for m in all_metrics])
            summary['sw_mae_my_mean'] = np.mean([np.mean(m.mae_my) for m in all_metrics])
            summary['sw_mae_my_std'] = np.std([np.mean(m.mae_my) for m in all_metrics])

            # h场守恒漂移（相对误差）
            summary['sw_cons_drift_h_mean'] = np.mean([np.mean(m.cons_drift_h) for m in all_metrics])
            summary['sw_cons_drift_h_std'] = np.std([np.mean(m.cons_drift_h) for m in all_metrics])

            # mx场守恒漂移（分开统计）
            all_mx_rel = [v for m in all_metrics for v in m.cons_drift_mx_rel]
            all_mx_abs = [v for m in all_metrics for v in m.cons_drift_mx_abs]
            summary['sw_cons_drift_mx_rel_mean'] = np.mean(all_mx_rel) if all_mx_rel else 0.0
            summary['sw_cons_drift_mx_rel_std'] = np.std(all_mx_rel) if all_mx_rel else 0.0
            summary['sw_cons_drift_mx_rel_count'] = len(all_mx_rel)
            summary['sw_cons_drift_mx_abs_mean'] = np.mean(all_mx_abs) if all_mx_abs else 0.0
            summary['sw_cons_drift_mx_abs_std'] = np.std(all_mx_abs) if all_mx_abs else 0.0
            summary['sw_cons_drift_mx_abs_count'] = len(all_mx_abs)

            # my场守恒漂移（分开统计）
            all_my_rel = [v for m in all_metrics for v in m.cons_drift_my_rel]
            all_my_abs = [v for m in all_metrics for v in m.cons_drift_my_abs]
            summary['sw_cons_drift_my_rel_mean'] = np.mean(all_my_rel) if all_my_rel else 0.0
            summary['sw_cons_drift_my_rel_std'] = np.std(all_my_rel) if all_my_rel else 0.0
            summary['sw_cons_drift_my_rel_count'] = len(all_my_rel)
            summary['sw_cons_drift_my_abs_mean'] = np.mean(all_my_abs) if all_my_abs else 0.0
            summary['sw_cons_drift_my_abs_std'] = np.std(all_my_abs) if all_my_abs else 0.0
            summary['sw_cons_drift_my_abs_count'] = len(all_my_abs)

            # h场越界统计
            if all_metrics[0].h_violation_rate_lower:
                summary['sw_h_viol_rate_mean'] = np.mean([np.mean(m.h_violation_rate_lower) for m in all_metrics])
                summary['sw_h_viol_rate_std'] = np.std([np.mean(m.h_violation_rate_lower) for m in all_metrics])
                all_h_cond = [v for m in all_metrics for v in m.h_cond_magnitude_lower if v > 0]
                summary['sw_h_cond_mag_mean'] = np.mean(all_h_cond) if all_h_cond else 0.0
                summary['sw_h_cond_mag_std'] = np.std(all_h_cond) if all_h_cond else 0.0

        return summary

    def _visualize_summary(self, all_metrics: List[EvaluationMetrics], summary: Dict,
                           output_dir: str, mode: str):
        """汇总可视化 (带误差棒)"""
        steps = range(summary['max_steps'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].errorbar(
            steps,
            summary['mae_mean_by_step'],
            yerr=summary['mae_std_by_step'],
            fmt='-o', capsize=3, markersize=2, linewidth=1
        )
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title(f'{mode.upper()} MAE (mean ± std)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].errorbar(
            steps,
            summary['rmse_mean_by_step'],
            yerr=summary['rmse_std_by_step'],
            fmt='-o', capsize=3, markersize=2, linewidth=1, color='orange'
        )
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title(f'{mode.upper()} RMSE (mean ± std)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].errorbar(
            steps,
            summary['cons_drift_mean_by_step'],
            yerr=summary['cons_drift_std_by_step'],
            fmt='-o', capsize=3, markersize=2, linewidth=1, color='green'
        )
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Drift (Adaptive)')
        axes[1, 0].set_title(f'{mode.upper()} Conservation Drift (mean ± std)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, summary['viol_lower_mean_by_step'], 'b-', linewidth=2, label='Lower')
        axes[1, 1].plot(steps, summary['viol_upper_mean_by_step'], 'r-', linewidth=2, label='Upper')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Violation Rate (%)')
        axes[1, 1].set_title(f'{mode.upper()} Bound Violation (mean)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Test Set Summary ({summary["num_trajectories"]} trajectories)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{mode}_summary.png'), dpi=150)
        plt.close()

    def _create_combined_gallery_image(self, vis_files: List[Path], output_dir: str,
                                       gallery_dir: str, mode: str):
        """创建组合大图"""
        from PIL import Image

        images = []
        labels = []
        for h5_file in vis_files:
            img_path = os.path.join(gallery_dir, f'{h5_file.stem}_{mode}_metrics.png')
            if os.path.exists(img_path):
                images.append(img_path)
                labels.append(h5_file.stem)

        if not images:
            return

        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        try:
            img_list = [Image.open(p) for p in images]
            w, h = img_list[0].size

            combined = Image.new('RGB', (cols * w, rows * h), color='white')

            for idx, (img, label) in enumerate(zip(img_list, labels)):
                row = idx // cols
                col = idx % cols
                combined.paste(img, (col * w, row * h))

            combined.save(os.path.join(gallery_dir, f'{mode}_combined_gallery.png'))

            for img in img_list:
                img.close()

        except Exception as e:
            print(f"创建组合大图失败: {e}")

    def _print_summary(self, summary: Dict):
        """打印汇总统计"""
        print("\n" + "=" * 70)
        print("测试集评估汇总")
        print("=" * 70)
        print(f"轨迹数量: {summary['num_trajectories']}")
        print(f"最大时间步: {summary['max_steps']}")
        print(f"\nMAE: {summary['mae_overall_mean']:.6e} ± {summary['mae_overall_std']:.6e}")
        print(f"RMSE: {summary['rmse_overall_mean']:.6e} ± {summary['rmse_overall_std']:.6e}")
        print(f"\n守恒漂移: mean={summary['cons_drift_mean']:.6e}, max={summary['cons_drift_max']:.6e}")

        if summary['viol_lower_mean'] > 0 or summary['viol_upper_mean'] > 0:
            print(f"\n越界率: 下界={summary['viol_lower_mean']:.4f}%, 上界={summary['viol_upper_mean']:.4f}%")

        print(f"\n值范围: [{summary['min_value_overall']:.6f}, {summary['max_value_overall']:.6f}]")

        # 浅水方程特殊统计
        if 'sw_mae_h_mean' in summary:
            print("\n" + "-" * 70)
            print("浅水方程分场统计:")
            print(f"  h场 MAE: {summary['sw_mae_h_mean']:.6e} ± {summary['sw_mae_h_std']:.6e}")
            print(f"  mx场 MAE: {summary['sw_mae_mx_mean']:.6e} ± {summary['sw_mae_mx_std']:.6e}")
            print(f"  my场 MAE: {summary['sw_mae_my_mean']:.6e} ± {summary['sw_mae_my_std']:.6e}")
            print("\n守恒漂移:")
            print(f"  h (质量) 相对漂移: {summary['sw_cons_drift_h_mean']:.6e}")
            print(
                f"  mx: 相对漂移={summary['sw_cons_drift_mx_rel_mean']:.6e} (n={summary['sw_cons_drift_mx_rel_count']}), "
                f"绝对漂移={summary['sw_cons_drift_mx_abs_mean']:.6e} (n={summary['sw_cons_drift_mx_abs_count']}, initial~0)")
            print(
                f"  my: 相对漂移={summary['sw_cons_drift_my_rel_mean']:.6e} (n={summary['sw_cons_drift_my_rel_count']}), "
                f"绝对漂移={summary['sw_cons_drift_my_abs_mean']:.6e} (n={summary['sw_cons_drift_my_abs_count']}, initial~0)")

        print("=" * 70)


def evaluate_model_on_test_set(
        model: torch.nn.Module,
        test_folder: str,
        dataset_type: str,
        output_dir: str,
        ndt: int = 1,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        device: torch.device = None,
        mode: str = 'both',
        visualize_trajectories: Optional[List[str]] = None,
        zero_threshold: float = 1e-12
) -> Dict:
    """
    便捷函数: 评估模型在测试集上的表现
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = UnifiedEvaluator(
        model=model,
        device=device,
        dataset_type=dataset_type,
        ndt=ndt,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_threshold=zero_threshold
    )

    results = {}

    if mode in ['onestep', 'both']:
        results['onestep'] = evaluator.evaluate_test_set_batch(
            test_folder,
            os.path.join(output_dir, 'onestep'),
            mode='onestep',
            visualize_trajectories=visualize_trajectories
        )

    if mode in ['rollout', 'both']:
        results['rollout'] = evaluator.evaluate_test_set_batch(
            test_folder,
            os.path.join(output_dir, 'rollout'),
            mode='rollout',
            visualize_trajectories=visualize_trajectories
        )

    return results
