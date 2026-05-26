"""
对流扩散 - 消融实验

可选模型完整列表（从中选取需要的进行消融实验）：

FluxNet模型系列:
- FluxNet_N_1D: 无约束 (原FluxNet_U_1D)
- FluxNet_P_1D: 正通量约束 (softplus)
- FluxNet_L_1D: 下界约束 (最优，c >= 0)

Baseline模型系列:
- CNN_1D_direct: 直接预测
- CNN_1D_residual: 残差预测
- CNN_1D_bound: 带下界约束 (softplus)
- CNN_1D_soft: 带软守恒损失

每种模型可选:
- onestep训练
- pushforward训练 (pf)
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.common.experiment_runner import (
    run_single_experiment, generate_summary_table, ModelConfig, TrainingConfig
)

# ============================================================================
# 完整可选模型名单 (从中选取进行实验)
# ============================================================================
ALL_AVAILABLE_MODELS = [
    # FluxNet系列 - onestep
    'FluxNet_N_1D_onestep',
    'FluxNet_P_1D_onestep',
    'FluxNet_L_1D_onestep',
    # FluxNet系列 - pushforward
    'FluxNet_N_1D_pf',
    'FluxNet_P_1D_pf',
    'FluxNet_L_1D_pf',
    # CNN Baseline系列 - onestep
    'CNN_1D_direct_onestep',
    'CNN_1D_residual_onestep',
    'CNN_1D_direct_soft_onestep',
    'CNN_1D_residual_soft_onestep',
    'CNN_1D_bound_onestep',
    'CNN_1D_bound_soft_onestep',
    # CNN Baseline系列 - pushforward
    'CNN_1D_direct_pf',
    'CNN_1D_residual_pf',
    'CNN_1D_direct_soft_pf',
    'CNN_1D_residual_soft_pf',
    'CNN_1D_bound_pf',
    'CNN_1D_bound_soft_pf',
]


SELECTED_MODELS = [
    'FluxNet_N_1D_onestep',              # 消融: 无约束
    'FluxNet_P_1D_onestep',              # 消融: 正通量
    'FluxNet_L_1D_onestep',         # 消融: L头 无pushforward
]



def _make_training_config(hparams: dict, use_pf: bool, soft_cons: float = 0.0) -> TrainingConfig:
    """
    创建TrainingConfig的辅助函数，统一处理loss_weight_mode

    使用方法:
        1. 默认使用自适应损失权重 (loss_weight_mode='adaptive')
        2. 如需手动指定，设置 hparams['loss_weight_mode'] = 'manual'
        3. 然后在 hparams['loss_weights'] 中指定各损失权重
           例如: {'p_loss': 1.0, 'stability_loss': 0.5, 'cons_loss': 0.1}
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
        - 可用损失项: p_loss (预测), stability_loss (稳定性), cons_loss (守恒)
    """
    use_pf = '_pf' in model_name

    # FluxNet系列
    if model_name.startswith('FluxNet_N_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_N_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size']
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_P_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_P_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size']
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_L_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_L_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    # CNN Baseline系列
    elif model_name.startswith('CNN_1D'):
        pred_mode = 'residual' if 'residual' in model_name else 'direct'
        soft_cons = hparams.get('soft_cons_weight', 0.1) if 'soft' in model_name else 0.0
        bound_mode = 'lower' if 'bound' in model_name else 'none'

        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='CNN_Baseline_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                prediction_mode=pred_mode,
                bound_mode=bound_mode,
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf, soft_cons)
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/convection_diffusion/ablation_onestep"
    train_folder = "FluxNet/dataset/convection_diffusion/train"
    val_folder = "FluxNet/dataset/convection_diffusion/val"
    test_folder = "FluxNet/dataset/convection_diffusion/test"

    # ========================================================================
    # 超参数配置
    # ========================================================================
    hparams = {
        # 模型架构参数
        'base_channels': 16,
        'num_blocks': 4,
        'kernel_size': 3,
        'neighborhood_size': 3,
        # 训练参数
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        # ================================================================
        # 损失权重配置
        # ================================================================
        'loss_weight_mode': 'adaptive',
        'soft_cons_weight': 0.1,
        'loss_weights': {
            'p_loss': 1.0,
            'stability_loss': 0.5,
            'cons_loss': 0.1,
        },
    }

    # ========================================================================
    # 实验控制
    # ========================================================================
    gpu_id = 0
    seed = 42
    run_training = True
    run_evaluation = True
    evaluate_mode = 'both'
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

    print(f"\n将运行 {len(ablation_experiments)} 个消融实验:")
    for exp in ablation_experiments:
        print(f"  - {exp['name']}")
    print()

    # ========================================================================
    # 运行实验
    # ========================================================================
    results = []

    for i, exp in enumerate(ablation_experiments):
        print(f"\n{'#'*80}")
        print(f"# 消融实验 {i+1}/{len(ablation_experiments)}: {exp['name']}")
        print(f"{'#'*80}\n")

        try:
            result = run_single_experiment(
                model_config=exp['model_config'],
                training_config=exp['training_config'],
                dataset_type='convection_diffusion',
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
    generate_summary_table(save_path, ablation_experiments, "ablation_summary.md")
    print(f"\n所有消融实验完成! 结果保存在: {save_path}")


if __name__ == "__main__":
    main()
