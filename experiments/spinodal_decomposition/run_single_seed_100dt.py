"""
调幅分解 - 消融实验 (ndt=1)

短时预测：ndt=1 意味着预测下一帧

推荐模型:
- FluxNet_D: 双界约束 (0 <= phi <= 1)
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.common.experiment_runner import (
    run_single_experiment, generate_summary_table, ModelConfig, TrainingConfig
)

# ============================================================================
# 完整可选模型名单
# ============================================================================
ALL_AVAILABLE_MODELS = [
    # FluxNet系列 - onestep
    'FluxNet_N_onestep',
    'FluxNet_P_onestep',
    'FluxNet_L_onestep',
    'FluxNet_D_onestep',
    # FluxNet系列 - pushforward
    'FluxNet_N_pf',
    'FluxNet_P_pf',
    'FluxNet_L_pf',
    'FluxNet_D_pf',
    # CNN Baseline系列 - onestep
    'CNN_2D_direct_onestep',
    'CNN_2D_residual_onestep',
    'CNN_2D_direct_soft_onestep',
    'CNN_2D_residual_soft_onestep',
    'CNN_2D_bound_onestep',
    'CNN_2D_bound_soft_onestep',
    # CNN Baseline系列 - pushforward
    'CNN_2D_direct_pf',
    'CNN_2D_residual_pf',
    'CNN_2D_direct_soft_pf',
    'CNN_2D_residual_soft_pf',
    'CNN_2D_bound_pf',
    'CNN_2D_bound_soft_pf',
]

# ============================================================================
# 实际运行的模型列表
# ============================================================================
SELECTED_MODELS = [
    # FluxNet系列
    # 'FluxNet_D_onestep',
    'FluxNet_D_pf',
    # CNN Baseline系列
    # 'CNN_2D_bound_soft_onestep',
]


def _make_training_config(hparams: dict, use_pf: bool, soft_cons: float = 0.0,
                          dcl_weight: float = 0.1) -> TrainingConfig:
    """
    创建TrainingConfig的辅助函数，统一处理loss_weight_mode

    Args:
        hparams: 超参数字典
        use_pf: 是否使用pushforward训练
        soft_cons: 软守恒损失权重 (仅baseline使用)
        dcl_weight: DCL损失权重 (仅FluxNet-D使用)

    Returns:
        TrainingConfig实例

    使用方法:
        1. 默认使用自适应损失权重 (loss_weight_mode='adaptive')
        2. 如需手动指定，设置 hparams['loss_weight_mode'] = 'manual'
        3. 然后在 hparams['loss_weights'] 中指定各损失权重
           例如: {'p_loss': 1.0, 'dcl_loss': 0.1, 'stability_loss': 0.5}
    """
    return TrainingConfig(
        num_epochs=hparams['num_epochs'],
        batch_size=hparams['batch_size'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        ndt=hparams['ndt'],
        num_workers=hparams['num_workers'],
        use_pushforward=use_pf,
        unroll_steps=hparams['unroll_steps'],
        soft_conservation_weight=soft_cons,
        dcl_weight=dcl_weight,
        loss_weight_mode=hparams.get('loss_weight_mode', 'adaptive'),
        loss_weights=hparams.get('loss_weights', {}),
    )


def get_experiment_config(model_name: str, hparams: dict) -> dict:
    """
    根据模型名称生成实验配置

    损失权重使用说明:
        - 默认使用自适应损失权重 (loss_weight_mode='adaptive')
        - 手动指定时，设置 hparams['loss_weight_mode'] = 'manual'
        - 然后在 hparams['loss_weights'] 中指定各损失权重
        - 对于FluxNet-D: 可用 p_loss, dcl_loss, stability_loss
        - 对于baseline: 可用 p_loss, cons_loss, stability_loss
    """
    use_pf = '_pf' in model_name

    # FluxNet系列
    if model_name.startswith('FluxNet_N_') and '1D' not in model_name:
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_N',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size']
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_P_') and '1D' not in model_name:
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_P',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size']
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_L_') and '1D' not in model_name:
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_L',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_D_') and '1D' not in model_name:
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': _make_training_config(
                hparams, use_pf,
                dcl_weight=hparams.get('dcl_weight', 0.1)
            )
        }

    # CNN Baseline系列
    elif model_name.startswith('CNN_2D'):
        pred_mode = 'residual' if 'residual' in model_name else 'direct'
        soft_cons = hparams.get('soft_cons_weight', 0.1) if 'soft' in model_name else 0.0
        bound_mode = 'double' if 'bound' in model_name else 'none'

        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='CNN_Baseline_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                prediction_mode=pred_mode,
                bound_mode=bound_mode,
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': _make_training_config(hparams, use_pf, soft_cons)
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    # ========================================================================
    # 路径配置
    # ========================================================================
    save_path = "FluxNet/results/spinodal_decomposition/ablation_100dt"
    train_folder = "FluxNet/dataset/spinodal_decomposition/train"
    val_folder = "FluxNet/dataset/spinodal_decomposition/val"
    test_folder = "FluxNet/dataset/spinodal_decomposition/test"

    # ========================================================================
    # 超参数配置 - ndt=1 (短时预测，对应10dt)
    # ========================================================================
    hparams = {
        # 模型架构参数
        'base_channels': 32,
        'num_blocks': 4,
        'kernel_size': 5,
        'neighborhood_size': 5,  # 小邻域适合小时间步长
        # 训练参数
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 10,  # 短时预测，对应10dt
        'num_workers': 4,
        'unroll_steps': 2,
        # ================================================================
        # 损失权重配置
        # ================================================================
        # 损失权重模式: 'adaptive' (自适应) 或 'manual' (手动指定)
        # 'loss_weight_mode': 'adaptive',
        'loss_weight_mode': 'manual',
        # DCL损失权重 (用于FluxNet-D)
        'dcl_weight': 0.5,
        # 软守恒损失权重 (用于baseline)
        'soft_cons_weight': 0.5,
        # 手动指定时的损失权重 (当loss_weight_mode='manual'时生效)
        # 可用损失项: p_loss (预测), dcl_loss (对偶一致性), stability_loss (稳定性), cons_loss (守恒)
        'loss_weights': {
            'p_loss': 0.5,
            'dcl_loss': 0.5,
            'stability_loss': 0.5,
        },
    }

    # ========================================================================
    # 实验控制
    # ========================================================================
    gpu_id = 3
    seed = 42
    run_training = False
    run_evaluation = True
    # evaluate_mode = 'both'
    evaluate_mode = 'rollout'
    # visualize_trajectories = None
    visualize_trajectories = 'all'

    # ========================================================================
    # 生成实验配置
    # ========================================================================
    ablation_experiments = []
    for model_name in SELECTED_MODELS:
        if model_name in ALL_AVAILABLE_MODELS:
            try:
                exp_config = get_experiment_config(model_name, hparams)
                ablation_experiments.append(exp_config)
            except ValueError as e:
                print(f"跳过模型 {model_name}: {e}")
        else:
            print(f"警告: 模型 '{model_name}' 不在可用列表中")

    print(f"\n[ndt=1] 将运行 {len(ablation_experiments)} 个消融实验:")
    for exp in ablation_experiments:
        print(f"  - {exp['name']}")
    print()

    # ========================================================================
    # 运行实验
    # ========================================================================
    results = []

    for i, exp in enumerate(ablation_experiments):
        print(f"\n{'#'*80}")
        print(f"# 消融实验 {i+1}/{len(ablation_experiments)}: {exp['name']} (ndt=1)")
        print(f"{'#'*80}\n")

        try:
            result = run_single_experiment(
                model_config=exp['model_config'],
                training_config=exp['training_config'],
                dataset_type='spinodal_decomposition',
                train_folder=train_folder,
                val_folder=val_folder,
                test_folder=test_folder,
                save_path=save_path,
                experiment_name=exp['name'],
                gpu_id=gpu_id,
                run_training=run_training,
                run_evaluation=run_evaluation,
                seed=seed,
                evaluate_mode=evaluate_mode,
                visualize_trajectories=visualize_trajectories
            )
            results.append(result)
        except Exception as e:
            print(f"实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # 生成汇总表格
    # ========================================================================
    print("\n" + "="*80)
    generate_summary_table(save_path, ablation_experiments, "ablation_summary_ndt1.md")
    print(f"\n所有消融实验完成! 结果保存在: {save_path}")


if __name__ == "__main__":
    main()
