"""
通用实验运行器

提供统一的训练、评估、可视化流程
用于超参数实验和消融实验
"""

import os
import sys
import json
import torch
import random
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models import *
from src.training import train_model, TrainingConfig, ModelConfig
from src.evaluation import evaluate_model_on_test_set


def set_seed(seed: int = 42):
    """设置随机种子保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_config: ModelConfig, dataset_type: str) -> torch.nn.Module:
    """
    根据配置创建模型

    Args:
        model_config: 模型配置
        dataset_type: 数据集类型
    """
    model_type = model_config.model_type
    config_dict = model_config.to_dict()

    # 根据数据集确定输入通道数
    if dataset_type == 'convection_diffusion':
        in_channels = 2  # c + u
    elif dataset_type == 'traffic_flow':
        in_channels = 2  # rho + vmax
    elif dataset_type == 'shallow_water':
        in_channels = 3  # h + mx + my
    elif dataset_type == 'spinodal_decomposition':
        in_channels = 1  # phi
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # 创建模型 - 支持新命名和旧命名
    # 1D FluxNet models
    if model_type == 'FluxNet_N_1D':
        return FluxNet_N_1D(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_P_1D':
        return FluxNet_P_1D(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_L_1D':
        return FluxNet_L_1D(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_D_1D':
        return FluxNet_D_1D(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_U_1D':
        from src.models import FluxNet_U_1D
        return FluxNet_U_1D(in_channels=in_channels, **config_dict)

    # 2D FluxNet models
    elif model_type in ['FluxNet_N', 'FluxNet_U']:
        return FluxNet_N(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_P':
        return FluxNet_P(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_L':
        return FluxNet_L(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxNet_D':
        return FluxNet_D(in_channels=in_channels, **config_dict)

    # Shallow water models
    elif model_type == 'FluxNet_SW_2D':
        return FluxNet_SW_2D(**config_dict)

    elif model_type == 'FluxNet_SW_Baseline':
        return FluxNet_SW_Baseline(**config_dict)

    elif model_type == 'FNO_SW':
        if FNO_SW is None:
            raise ImportError("FNO_SW not available")
        return FNO_SW(**config_dict)

    # Shallow water baselines with projection
    elif model_type == 'FNO_SW_Proj':
        from src.models import FNO_SW_Proj
        if FNO_SW_Proj is None:
            raise ImportError("FNO_SW_Proj not available")
        return FNO_SW_Proj(**config_dict)

    elif model_type == 'CNN_SW_Proj':
        from src.models import CNN_SW_Proj
        if CNN_SW_Proj is None:
            raise ImportError("CNN_SW_Proj not available")
        return CNN_SW_Proj(**config_dict)

    # 1D FNO models
    elif model_type == 'FNO_1D':
        from src.models import FNO_1D
        if FNO_1D is None:
            raise ImportError("FNO_1D not available")
        return FNO_1D(in_channels=in_channels, out_channels=1, **config_dict)

    elif model_type == 'FNO_FluxD_1D':
        from src.models import FNO_FluxD_1D
        if FNO_FluxD_1D is None:
            raise ImportError("FNO_FluxD_1D not available")
        return FNO_FluxD_1D(in_channels=in_channels, **config_dict)

    # FNO with FluxLAP head (shallow water ablation)
    elif model_type == 'FNO_FluxLAP':
        from src.models import FNO_FluxLAP
        if FNO_FluxLAP is None:
            raise ImportError("FNO_FluxLAP not available")
        return FNO_FluxLAP(**config_dict)

    # Dirichlet boundary models (traffic flow only)
    elif model_type == 'FluxNet_D_Dirichlet_1D':
        from src.models import FluxNet_D_Dirichlet_1D
        if FluxNet_D_Dirichlet_1D is None:
            raise ImportError("FluxNet_D_Dirichlet_1D not available")
        return FluxNet_D_Dirichlet_1D(in_channels=in_channels, **config_dict)

    elif model_type == 'FluxGNN_1D':
        from src.models import FluxGNN_1D
        if FluxGNN_1D is None:
            raise ImportError("FluxGNN_1D not available")
        return FluxGNN_1D(in_channels=in_channels, out_channels=1, **config_dict)

    elif model_type == 'FluxGNN_D_1D':
        from src.models import FluxGNN_D_1D
        if FluxGNN_D_1D is None:
            raise ImportError("FluxGNN_D_1D not available")
        return FluxGNN_D_1D(in_channels=in_channels, **config_dict)

    # CNN baselines
    elif model_type == 'CNN_Baseline_1D':
        return CNN_Baseline_1D(in_channels=in_channels, out_channels=1, **config_dict)

    elif model_type == 'CNN_Baseline_2D':
        return CNN_Baseline_2D(in_channels=in_channels, out_channels=1, **config_dict)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_experiment_name(model_config: ModelConfig, training_config: TrainingConfig) -> str:
    """生成实验名称"""
    parts = [model_config.model_type]
    parts.append(f"c{model_config.base_channels}")
    parts.append(f"b{model_config.num_blocks}")
    parts.append(f"k{model_config.kernel_size}")

    if 'FluxNet' in model_config.model_type and 'Baseline' not in model_config.model_type:
        parts.append(f"n{model_config.neighborhood_size}")

    parts.append(f"ndt{training_config.ndt}")

    if training_config.use_pushforward:
        parts.append("pf")

    if training_config.soft_conservation_weight > 0:
        parts.append("soft")

    return "_".join(parts)


def get_dataset_bounds(dataset_type: str) -> Dict:
    """
    获取数据集的固有边界

    这些是数据集的物理边界，与模型无关
    用于统计越界率
    """
    bounds = {
        'convection_diffusion': {'lower_bound': 0.0, 'upper_bound': None},  # c >= 0
        'traffic_flow': {'lower_bound': 0.0, 'upper_bound': 1.0},  # 0 <= rho <= 1
        'shallow_water': {'lower_bound': 0.0, 'upper_bound': None},  # h >= 0 (仅h场)
        'spinodal_decomposition': {'lower_bound': 0.0, 'upper_bound': 1.0},  # 0 <= phi <= 1
    }
    return bounds.get(dataset_type, {'lower_bound': None, 'upper_bound': None})


def run_single_experiment(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_type: str,
    train_folder: str,
    val_folder: str,
    test_folder: str,
    save_path: str,
    experiment_name: Optional[str] = None,
    gpu_id: int = 0,
    run_training: bool = True,
    run_evaluation: bool = True,
    seed: int = 42,
    evaluate_mode: str = 'both',
    visualize_trajectories: Optional[str] = None
) -> Dict:
    """
    运行单个实验

    Args:
        model_config: 模型配置
        training_config: 训练配置
        dataset_type: 数据集类型
        train_folder: 训练数据目录 (绝对路径)
        val_folder: 验证数据目录 (绝对路径)
        test_folder: 测试数据目录 (绝对路径)
        save_path: 结果保存根目录
        experiment_name: 实验名称 (None则自动生成)
        gpu_id: GPU编号
        run_training: 是否运行训练
        run_evaluation: 是否运行评估
        seed: 随机种子
        evaluate_mode: 评估模式 ('onestep', 'rollout', 'both')
        visualize_trajectories: 可视化轨迹 ('all', None, 或具体列表)

    Returns:
        实验结果字典
    """
    # 设置随机种子
    set_seed(seed)

    # 生成实验名称
    if experiment_name is None:
        experiment_name = get_experiment_name(model_config, training_config)

    print("=" * 80)
    print(f"实验: {experiment_name}")
    print(f"数据集: {dataset_type}")
    print(f"模型: {model_config.model_type}")
    print("=" * 80)

    # 设置设备
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    result_dir = os.path.join(save_path, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果保存至: {result_dir}")

    # 创建模型
    model = create_model(model_config, dataset_type)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 获取数据集固有边界
    dataset_bounds = get_dataset_bounds(dataset_type)

    # 保存配置
    config_data = {
        'experiment_name': experiment_name,
        'dataset_type': dataset_type,
        'model_config': model_config.__dict__,
        'training_config': training_config.to_dict(),
        'seed': seed,
        'param_count': param_count,
        'dataset_bounds': dataset_bounds,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    results = {'experiment_name': experiment_name}

    # ========== 训练 ==========
    if run_training:
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        train_results = train_model(
            model=model,
            dataset_type=dataset_type,
            train_folder=train_folder,
            val_folder=val_folder,
            result_dir=result_dir,
            config=training_config,
            device=device,
            num_workers=training_config.num_workers
        )

        results['training'] = {
            'best_loss': train_results['best_loss'],
            'total_time': train_results['total_time']
        }

    # ========== 评估 ==========
    if run_evaluation:
        print("\n" + "=" * 60)
        print("开始评估")
        print("=" * 60)

        # 加载最优模型
        best_model_path = os.path.join(result_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"已加载最优模型: {best_model_path}")
        else:
            print("警告: 未找到最优模型，使用当前模型状态")

        model.eval()

        # 使用数据集固有边界进行评估
        eval_output_dir = os.path.join(result_dir, 'evaluation')

        eval_results = evaluate_model_on_test_set(
            model=model,
            test_folder=test_folder,
            dataset_type=dataset_type,
            output_dir=eval_output_dir,
            ndt=training_config.ndt,
            lower_bound=dataset_bounds['lower_bound'],
            upper_bound=dataset_bounds['upper_bound'],
            device=device,
            mode=evaluate_mode,
            visualize_trajectories=visualize_trajectories
        )

        results['evaluation'] = {}
        if 'onestep' in eval_results:
            results['evaluation']['onestep'] = {
                'mae': eval_results['onestep']['mae_overall_mean'],
                'mae_std': eval_results['onestep']['mae_overall_std'],
                'rmse': eval_results['onestep']['rmse_overall_mean'],
                'cons_drift_mean': eval_results['onestep']['cons_drift_mean'],
                'cons_drift_max': eval_results['onestep']['cons_drift_max'],
                'viol_lower': eval_results['onestep']['viol_lower_mean'],
                'viol_upper': eval_results['onestep']['viol_upper_mean'],
            }
        # 改这个地方，不要再全局了，而是最后一帧的rollout误差
        if 'rollout' in eval_results:
            results['evaluation']['rollout'] = {
                'mae': eval_results['rollout']['mae_overall_mean'],
                'mae_std': eval_results['rollout']['mae_overall_std'],
                'rmse': eval_results['rollout']['rmse_overall_mean'],
                'cons_drift_mean': eval_results['rollout']['cons_drift_mean'],
                'cons_drift_max': eval_results['rollout']['cons_drift_max'],
                'viol_lower': eval_results['rollout']['viol_lower_mean'],
                'viol_upper': eval_results['rollout']['viol_upper_mean'],
            }

        # 评估时才保存结果 不要改！！！
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存精简结果用于汇总
        import joblib
        joblib.dump(results, os.path.join(result_dir, 'results.pkl'))

    print("\n" + "=" * 60)
    print(f"实验完成: {experiment_name}")
    print("=" * 60)

    return results


def generate_summary_table(save_path: str, experiments: List[Dict], output_file: str = "summary.md"):
    """
    生成实验汇总表格 (Markdown格式)

    包含:
    - Rollout表格: @T=1.0时间点误差 (rollout最终时刻，而非全局平均)
    - 守恒性能和越界统计 (含方差)
    - 条件越界幅度 (Conditional Mean OOB Magnitude)
    - 浅水方程三场分别统计
    """
    import joblib

    results = []
    for exp in experiments:
        exp_name = exp.get('name', get_experiment_name(exp['model_config'], exp['training_config']))
        result_file = os.path.join(save_path, exp_name, 'results.pkl')

        if os.path.exists(result_file):
            result = joblib.load(result_file)
            result['config'] = exp
            results.append(result)

            # 尝试读取详细评估结果
            eval_dir = os.path.join(save_path, exp_name, 'evaluation')
            for mode in ['onestep', 'rollout']:
                summary_file = os.path.join(eval_dir, mode, f'test_set_summary_{mode}.json')
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        result[f'{mode}_detailed'] = json.load(f)

    if not results:
        print("没有找到任何实验结果")
        return

    valid_results = [r for r in results if 'evaluation' in r]
    if not valid_results:
        print("没有找到有效的评估结果")
        return

    # 找出最优结果
    rollout_results = [r for r in valid_results if 'rollout' in r['evaluation']]
    best_rollout_mae = min(r.get('rollout_detailed', {}).get('mae_at_T1.0', r['evaluation']['rollout'].get('mae', float('inf')))
                          for r in rollout_results) if rollout_results else None

    # ========== 生成Markdown内容 ==========
    md_content = f"""# 消融实验汇总

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Rollout 评估结果 (@T=1.0, 最终时刻)

**注意**: 所有rollout指标均为T=1.0时刻的值，而非全局演化平均值。

| 模型 | MAE@T=1.0 (mean±std) | RMSE@T=1.0 | 守恒漂移@T=1.0 (mean±std) | 最大守恒漂移 |
|------|---------------------|------------|--------------------------|------------|
"""

    for r in valid_results:
        exp_name = r['experiment_name']
        rollout = r['evaluation'].get('rollout', {})
        detailed = r.get('rollout_detailed', {})

        if rollout:
            # 使用@T=1.0的值
            mae_100 = detailed.get('mae_at_T1.0', rollout.get('mae', 0))
            mae_100_std = detailed.get('mae_at_T1.0_std', rollout.get('mae_std', 0))
            rmse_100 = detailed.get('rmse_at_T1.0', rollout.get('rmse', 0))
            cons_100 = detailed.get('cons_drift_at_T1.0', rollout.get('cons_drift_mean', 0))
            cons_100_std = detailed.get('cons_drift_at_T1.0_std', rollout.get('cons_drift_std', 0))
            cons_max = rollout.get('cons_drift_max', 0)

            mae_str = f"{mae_100:.2e}±{mae_100_std:.2e}"
            if best_rollout_mae and abs(mae_100 - best_rollout_mae) < 1e-10:
                mae_str = f"**{mae_str}**"

            md_content += f"| {exp_name} | {mae_str} | {rmse_100:.2e} | {cons_100:.2e}±{cons_100_std:.2e} | {cons_max:.2e} |\n"

    md_content += f"""
---

## 2. 越界统计 (Rollout@T=1.0)

| 模型 | 下界越界率(mean±std) | 上界越界率(mean±std) | 条件越界幅度(下) | 条件越界幅度(上) | 预测值范围 |
|------|---------------------|---------------------|-----------------|-----------------|-----------|
"""

    for r in valid_results:
        exp_name = r['experiment_name']
        rollout = r['evaluation'].get('rollout', {})
        detailed = r.get('rollout_detailed', {})

        if rollout:
            # @T=1.0时刻的越界率
            viol_l = detailed.get('viol_lower_at_T1.0', rollout.get('viol_lower', 0))
            viol_l_std = detailed.get('viol_lower_at_T1.0_std', rollout.get('viol_lower_std', 0))
            viol_u = detailed.get('viol_upper_at_T1.0', rollout.get('viol_upper', 0))
            viol_u_std = detailed.get('viol_upper_at_T1.0_std', rollout.get('viol_upper_std', 0))

            viol_l_str = f"{viol_l:.2f}%±{viol_l_std:.2f}%"
            viol_u_str = f"{viol_u:.2f}%±{viol_u_std:.2f}%"

            # 条件越界幅度 (如果有)
            cond_mag_l = detailed.get('cond_magnitude_lower_mean', 0)
            cond_mag_l_std = detailed.get('cond_magnitude_lower_std', 0)
            cond_mag_u = detailed.get('cond_magnitude_upper_mean', 0)
            cond_mag_u_std = detailed.get('cond_magnitude_upper_std', 0)
            cond_l_str = f"{cond_mag_l:.2e}±{cond_mag_l_std:.2e}" if cond_mag_l > 0 else "N/A"
            cond_u_str = f"{cond_mag_u:.2e}±{cond_mag_u_std:.2e}" if cond_mag_u > 0 else "N/A"

            min_val = detailed.get('min_value_overall', 0)
            max_val = detailed.get('max_value_overall', 1)
            range_str = f"[{min_val:.4f}, {max_val:.4f}]"

            md_content += f"| {exp_name} | {viol_l_str} | {viol_u_str} | {cond_l_str} | {cond_u_str} | {range_str} |\n"

    # 检查是否有浅水方程的三场统计
    has_sw_stats = any(r.get('rollout_detailed', {}).get('sw_mae_h_mean') for r in valid_results)
    if has_sw_stats:
        md_content += f"""
---

## 3. 浅水方程三场分别统计 (Rollout)

| 模型 | h场MAE (mean±std) | mx场MAE (mean±std) | my场MAE (mean±std) |
|------|------------------|-------------------|-------------------|
"""
        for r in valid_results:
            exp_name = r['experiment_name']
            detailed = r.get('rollout_detailed', {})
            if detailed.get('sw_mae_h_mean'):
                h_mae = f"{detailed.get('sw_mae_h_mean', 0):.2e}±{detailed.get('sw_mae_h_std', 0):.2e}"
                mx_mae = f"{detailed.get('sw_mae_mx_mean', 0):.2e}±{detailed.get('sw_mae_mx_std', 0):.2e}"
                my_mae = f"{detailed.get('sw_mae_my_mean', 0):.2e}±{detailed.get('sw_mae_my_std', 0):.2e}"
                md_content += f"| {exp_name} | {h_mae} | {mx_mae} | {my_mae} |\n"

        md_content += f"""
### 浅水方程守恒漂移

| 模型 | h守恒漂移 (mean±std) | mx守恒漂移 (mean±std) | my守恒漂移 (mean±std) |
|------|---------------------|----------------------|----------------------|
"""
        for r in valid_results:
            exp_name = r['experiment_name']
            detailed = r.get('rollout_detailed', {})
            if detailed.get('sw_cons_drift_h_mean'):
                h_cons = f"{detailed.get('sw_cons_drift_h_mean', 0):.2e}±{detailed.get('sw_cons_drift_h_std', 0):.2e}"
                mx_cons = f"{detailed.get('sw_cons_drift_mx_abs_mean', 0)/100:.2e}±{detailed.get('sw_cons_drift_mx_abs_std', 0)/100:.2e}"
                my_cons = f"{detailed.get('sw_cons_drift_my_abs_mean', 0)/100:.2e}±{detailed.get('sw_cons_drift_my_abs_std', 0)/100:.2e}"
                md_content += f"| {exp_name} | {h_cons} | {mx_cons} | {my_cons} |\n"

        md_content += f"""
### 浅水方程h场越界统计

| 模型 | h下界越界率 (mean±std) | h条件越界幅度 (mean±std) |
|------|-----------------------|-------------------------|
"""
        for r in valid_results:
            exp_name = r['experiment_name']
            detailed = r.get('rollout_detailed', {})
            if detailed.get('sw_h_viol_rate_mean') is not None:
                h_viol = f"{detailed.get('sw_h_viol_rate_mean', 0):.2f}%±{detailed.get('sw_h_viol_rate_std', 0):.2f}%"
                h_cond = detailed.get('sw_h_cond_mag_mean', 0)
                h_cond_std = detailed.get('sw_h_cond_mag_std', 0)
                h_cond_str = f"{h_cond:.2e}±{h_cond_std:.2e}" if h_cond > 0 else "N/A"
                md_content += f"| {exp_name} | {h_viol} | {h_cond_str} |\n"

    md_content += f"""
---

## 说明

- **粗体**表示该指标的最优值
- 所有rollout指标均为**@T=1.0时刻**的值（最终时刻），而非全局演化平均值
- 误差格式: mean±std (跨测试轨迹)
- 守恒漂移: 相对守恒误差
- 条件越界幅度 (Conditional Mean OOB Magnitude): 仅统计越界点的平均越界量，用于衡量"出问题的点偏离了多少"

## 最优配置

- Rollout MAE@T=1.0最优: {f'{best_rollout_mae:.4e}' if best_rollout_mae else 'N/A'}
"""

    # 保存
    with open(os.path.join(save_path, output_file), 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"汇总表格已保存至: {os.path.join(save_path, output_file)}")


if __name__ == "__main__":
    print("这是通用实验运行器模块")
    print("请创建具体的实验脚本来使用此模块")
