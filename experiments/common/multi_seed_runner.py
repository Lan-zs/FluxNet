"""
多随机种子实验运行器

提供多随机种子训练+评估的通用框架：
- 随机种子循环训练+评估
- 自动跳过已完成实验
- 结果收集与统计汇总

目录结构: save_path/RandomSeed{seed}/model_name/ (训练+评估结果)
结果收集: 从 test_set_summary_rollout.json 读取各seed的结果，汇总均值和标准差
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Callable


# ============================================================================
# 不同数据集的指标收集配置
# ============================================================================
# 1D数据集 (traffic_flow, convection_diffusion): 基础指标
METRICS_1D = [
    'mae_overall_mean', 'mae_overall_std',
    'rmse_overall_mean', 'rmse_overall_std',
    'cons_drift_mean', 'cons_drift_std',
    'cons_drift_max',
    'viol_lower_mean', 'viol_lower_std',
    'viol_upper_mean', 'viol_upper_std',
]

# 1D数据集 + 最终时间点指标
METRICS_1D_WITH_FINAL = METRICS_1D + [
    'mae_at_T1.0', 'mae_at_T1.0_std',
    'rmse_at_T1.0', 'rmse_at_T1.0_std',
    'cons_drift_at_T1.0', 'cons_drift_at_T1.0_std',
    'viol_lower_at_T1.0', 'viol_lower_at_T1.0_std',
    'viol_upper_at_T1.0', 'viol_upper_at_T1.0_std',
    'cond_magnitude_lower_mean', 'cond_magnitude_upper_mean',
]

# shallow_water: 三场分别统计
METRICS_SW = [
    'mae_overall_mean', 'mae_overall_std',
    'rmse_overall_mean', 'rmse_overall_std',
    'cons_drift_mean', 'cons_drift_std',
    'cons_drift_max',
    'viol_lower_mean', 'viol_lower_std',
    'viol_upper_mean', 'viol_upper_std',
    # 浅水方程三场
    'sw_mae_h_mean', 'sw_mae_h_std',
    'sw_mae_mx_mean', 'sw_mae_mx_std',
    'sw_mae_my_mean', 'sw_mae_my_std',
    # 守恒漂移
    'sw_cons_drift_h_mean', 'sw_cons_drift_h_std',
    'sw_cons_drift_mx_rel_mean', 'sw_cons_drift_mx_rel_std',
    'sw_cons_drift_my_rel_mean', 'sw_cons_drift_my_rel_std',
    'sw_cons_drift_mx_abs_mean','sw_cons_drift_my_abs_mean',
    # 越界
    'sw_h_viol_rate_mean', 'sw_h_viol_rate_std',
    'sw_h_cond_mag_mean', 'sw_h_cond_mag_std',
]

# spinodal_decomposition: 与1D类似但2D
METRICS_SPINODAL = METRICS_1D_WITH_FINAL


def generate_seeds(num_seeds: int = 20, base_seed: int = 42) -> List[int]:
    """生成随机种子列表"""
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31)) for _ in range(num_seeds)]


def check_experiment_complete(save_path: str, seed: int, experiment_name: str,
                              evaluate_mode: str = 'rollout') -> bool:
    """
    检查实验是否已完成

    判断标准:
    1. 模型权重文件存在: save_path/RandomSeed{seed}/{experiment_name}/best_model.pt
    2. 评估结果文件存在: save_path/RandomSeed{seed}/{experiment_name}/evaluation/{mode}/test_set_summary_{mode}.json
    """
    seed_dir = os.path.join(save_path, f"RandomSeed{seed}", experiment_name)

    # 检查模型权重
    model_file = os.path.join(seed_dir, 'best_model.pt')
    if not os.path.exists(model_file):
        return False

    # 检查评估结果
    summary_file = os.path.join(seed_dir, 'evaluation', evaluate_mode,
                                f'test_set_summary_{evaluate_mode}.json')
    if not os.path.exists(summary_file):
        return False

    return True


def run_multi_seed_experiment(
    save_path: str,
    seeds: List[int],
    get_experiment_config_fn: Callable[[str, dict], dict],
    selected_models: List[str],
    hparams: dict,
    dataset_type: str,
    train_folder: str,
    val_folder: str,
    test_folder: str,
    gpu_id: int = 0,
    evaluate_mode: str = 'rollout',
    visualize_trajectories: Optional[str] = None,
    metrics_keys: List[str] = None,
    run_training: bool = True,
    run_evaluation: bool = True,
) -> Dict:
    """
    多随机种子实验运行器

    Args:
        save_path: 结果保存根目录 (如 FluxNet/results/convection_diffusion/multi_seed)
        seeds: 随机种子列表
        get_experiment_config_fn: 生成实验配置的函数 (来自run_ablation.py)
        selected_models: 要运行的模型列表
        hparams: 超参数字典
        dataset_type: 数据集类型
        train_folder: 训练数据目录
        val_folder: 验证数据目录
        test_folder: 测试数据目录
        gpu_id: GPU编号
        evaluate_mode: 评估模式 ('rollout')
        visualize_trajectories: 可视化轨迹 (None表示不可视化)
        metrics_keys: 要收集的指标列表
        run_training: 是否运行训练
        run_evaluation: 是否运行评估

    Returns:
        汇总结果字典
    """
    from experiments.common.experiment_runner import run_single_experiment

    if metrics_keys is None:
        metrics_keys = METRICS_1D_WITH_FINAL

    total_experiments = len(seeds) * len(selected_models)
    completed = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*80}")
    print(f"多随机种子实验")
    print(f"数据集: {dataset_type}")
    print(f"模型数量: {len(selected_models)}")
    print(f"种子数量: {len(seeds)}")
    print(f"总实验数: {total_experiments}")
    print(f"评估模式: {evaluate_mode}")
    print(f"结果保存: {save_path}")
    print(f"{'='*80}\n")

    for seed in seeds:
        seed_save_path = os.path.join(save_path, f"RandomSeed{seed}")

        for model_name in selected_models:
            exp_config = get_experiment_config_fn(model_name, hparams)

            # 检查是否已完成
            if check_experiment_complete(save_path, seed, exp_config['name'], evaluate_mode):
                print(f"[SKIP] Seed={seed}, Model={exp_config['name']} - 已完成")
                skipped += 1
                continue

            print(f"\n{'#'*80}")
            print(f"# Seed={seed}, Model={exp_config['name']}")
            print(f"# 进度: {completed + skipped + failed + 1}/{total_experiments}")
            print(f"{'#'*80}\n")

            try:
                result = run_single_experiment(
                    model_config=exp_config['model_config'],
                    training_config=exp_config['training_config'],
                    dataset_type=dataset_type,
                    train_folder=train_folder,
                    val_folder=val_folder,
                    test_folder=test_folder,
                    save_path=seed_save_path,
                    experiment_name=exp_config['name'],
                    gpu_id=gpu_id,
                    run_training=run_training,
                    run_evaluation=run_evaluation,
                    seed=seed,
                    evaluate_mode=evaluate_mode,
                    visualize_trajectories=visualize_trajectories,
                )
                completed += 1
            except Exception as e:
                print(f"[FAIL] Seed={seed}, Model={exp_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue

    print(f"\n{'='*80}")
    print(f"实验统计: 完成={completed}, 跳过={skipped}, 失败={failed}, 总计={total_experiments}")
    print(f"{'='*80}")

    # 收集结果
    aggregated = collect_results(
        save_path=save_path,
        seeds=seeds,
        selected_models=selected_models,
        get_experiment_config_fn=get_experiment_config_fn,
        hparams=hparams,
        evaluate_mode=evaluate_mode,
        metrics_keys=metrics_keys,
    )

    # 保存汇总结果
    summary_file = os.path.join(save_path, 'multi_seed_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n汇总结果已保存至: {summary_file}")

    # 打印汇总表格
    print_summary_table(aggregated, metrics_keys)

    return aggregated


def collect_results(
    save_path: str,
    seeds: List[int],
    selected_models: List[str],
    get_experiment_config_fn: Callable[[str, dict], dict],
    hparams: dict,
    evaluate_mode: str = 'rollout',
    metrics_keys: List[str] = None,
) -> Dict:
    """
    从各seed的结果中收集指标，计算均值和标准差

    Returns:
        {
            model_name: {
                'per_seed': {seed: {metric: value, ...}, ...},
                'mean': {metric: mean_value, ...},
                'std': {metric: std_value, ...},
                'num_seeds_completed': int,
                'missing_seeds': [seed1, seed2, ...],
            },
            ...
        }
    """
    if metrics_keys is None:
        metrics_keys = METRICS_1D_WITH_FINAL

    aggregated = {}

    for model_name in selected_models:
        exp_config = get_experiment_config_fn(model_name, hparams)
        experiment_name = exp_config['name']

        per_seed = {}
        missing_seeds = []

        for seed in seeds:
            summary_file = os.path.join(
                save_path, f"RandomSeed{seed}", experiment_name,
                'evaluation', evaluate_mode,
                f'test_set_summary_{evaluate_mode}.json'
            )

            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                per_seed[seed] = {}
                for key in metrics_keys:
                    if key in data:
                        per_seed[seed][key] = data[key]
                    else:
                        per_seed[seed][key] = None
                        print(f"[WARN] Seed={seed}, Model={experiment_name}: 缺少指标 '{key}'")
            else:
                missing_seeds.append(seed)
                print(f"[WARN] Seed={seed}, Model={experiment_name}: 缺少结果文件 {summary_file}")

        # 计算均值和标准差
        mean_dict = {}
        std_dict = {}

        for key in metrics_keys:
            values = [per_seed[s][key] for s in per_seed if per_seed[s][key] is not None]
            if values:
                mean_dict[key] = float(np.mean(values))
                std_dict[key] = float(np.std(values))
            else:
                mean_dict[key] = None
                std_dict[key] = None

        num_completed = len(per_seed)
        if missing_seeds:
            print(f"[WARN] Model={experiment_name}: {len(missing_seeds)}个seed缺少结果: {missing_seeds}")

        aggregated[experiment_name] = {
            'per_seed': per_seed,
            'mean': mean_dict,
            'std': std_dict,
            'num_seeds_completed': num_completed,
            'missing_seeds': missing_seeds,
        }

    return aggregated


def print_summary_table(aggregated: Dict, metrics_keys: List[str] = None):
    """打印多随机种子汇总表格"""
    if metrics_keys is None:
        metrics_keys = METRICS_1D_WITH_FINAL

    # 选择关键指标显示
    display_keys = [k for k in metrics_keys if 'mean' in k and 'by_step' not in k]

    print(f"\n{'='*100}")
    print("多随机种子实验汇总 (均值±标准差)")
    print(f"{'='*100}")

    # 表头
    header = f"| {'Model':<30} | {'Seeds':>5} |"
    for key in display_keys:
        short_name = key.replace('_overall_', '_').replace('_mean', '').replace('_at_T1.0', '@T1')
        header += f" {short_name:>12} |"
    print(header)
    print("-" * len(header))

    # 数据行
    for model_name, data in aggregated.items():
        row = f"| {model_name:<30} | {data['num_seeds_completed']:>5} |"
        for key in display_keys:
            m = data['mean'].get(key)
            s = data['std'].get(key)
            if m is not None and s is not None:
                row += f" {m:.4e}±{s:.2e} |"
            else:
                row += f" {'N/A':>12} |"
        print(row)

    print(f"{'='*100}\n")