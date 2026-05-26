"""
交通流 - 消融实验

可选模型完整列表（从中选取需要的进行消融实验）：

FluxNet模型系列:
- FluxNet_N_1D: 无约束 (原FluxNet_U_1D)
- FluxNet_P_1D: 正通量约束 (softplus)
- FluxNet_L_1D: 下界约束
- FluxNet_D_1D: 双界约束 (最优)

Baseline模型系列:
- CNN_1D_direct: 直接预测
- CNN_1D_residual: 残差预测
- CNN_1D_bound: 带双界约束 (sigmoid)
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
    'FluxNet_N_1D_onestep',   # No constraint (deprecated)
    'FluxNet_P_1D_onestep',   # Positive flux (softplus)
    'FluxNet_L_1D_onestep',   # Lower bound only
    'FluxNet_D_1D_onestep',   # Double bound (RECOMMENDED for traffic flow)
    'FluxNet_U_1D_onestep',   # Upper bound only (ablation)
    'FluxNet_D_1D_no_dcl_onestep',  # D without DCL loss (ablation)
    # FluxNet系列 - pushforward
    'FluxNet_N_1D_pf',
    'FluxNet_P_1D_pf',
    'FluxNet_L_1D_pf',
    'FluxNet_D_1D_pf',
    'FluxNet_U_1D_pf',
    'FluxNet_D_1D_no_dcl_pf',  # D without DCL loss (ablation)
    # FNO系列 - onestep
    'FNO_1D_onestep',         # Standard FNO
    'FNO_1D_soft_onestep',    # FNO + soft conservation loss
    'FNO_FluxD_1D_onestep',   # FNO with FluxNet-D head (FNO backbone + flux head)
    # FNO系列 - pushforward
    'FNO_1D_pf',
    'FNO_1D_soft_pf',         # FNO + soft conservation loss
    'FNO_FluxD_1D_pf',
    # CNN Baseline系列 - onestep
    'CNN_1D_direct_onestep',
    'CNN_1D_residual_onestep',
    'CNN_1D_direct_soft_onestep',
    'CNN_1D_residual_soft_onestep',
    'CNN_1D_bound_onestep',         # sigmoid bound
    'CNN_1D_bound_soft_onestep',    # sigmoid bound + soft conservation
    # CNN Baseline系列 - pushforward
    'CNN_1D_direct_pf',
    'CNN_1D_residual_pf',
    'CNN_1D_direct_soft_pf',
    'CNN_1D_residual_soft_pf',
    'CNN_1D_bound_pf',
    'CNN_1D_bound_soft_pf',
]

# ============================================================================
# 实际运行的模型列表
# 根据issue126.md指定的基线对比和消融实验
# ============================================================================
SELECTED_MODELS = [
    # ===== 我的方法 (FluxNet-D) =====
    'FluxNet_D_1D_pf',              # 最优方法: 双界约束 + pushforward

    # ===== 基线对比 =====
    'FNO_1D_pf',                    # FNO基线
    'FNO_1D_soft_pf',               # FNO + 软守恒损失
    'CNN_1D_residual_pf',           # CNN基线 (残差预测)
    'CNN_1D_residual_soft_pf',      # CNN + 软守恒损失
    'CNN_1D_bound_soft_pf',         # CNN + 有界(sigmoid) + 软守恒损失

    # ===== 消融实验 =====
    'FluxNet_P_1D_pf',              # 消融: P头 (正通量)
    'FluxNet_L_1D_pf',              # 消融: L头 (仅下界)
    'FluxNet_U_1D_pf',              # 消融: U头 (仅上界)
    'FluxNet_D_1D_no_dcl_pf',       # 消融: D头 无对偶一致性损失
    'FluxNet_D_1D_onestep',         # 消融: D头 无pushforward
    'FNO_FluxD_1D_pf',              # 消融: FNO backbone + D头
]


def get_experiment_config(model_name: str, hparams: dict) -> dict:
    """根据模型名称生成实验配置"""
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
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
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
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
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
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
        }

    # FluxNet_D_1D without DCL loss (ablation)
    elif model_name.startswith('FluxNet_D_1D_no_dcl'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_D_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps'],
                dcl_weight=0.0,  # 禁用DCL损失
                loss_weight_mode='manual',
                loss_weights=hparams.get('loss_weights', {'p_loss': 1.0, 'stability_loss': 0.5})
            )
        }

    elif model_name.startswith('FluxNet_D_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_D_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps'],
                dcl_weight=hparams.get('dcl_weight', 0.1),
                loss_weight_mode=hparams.get('loss_weight_mode', 'adaptive'),
                loss_weights=hparams.get('loss_weights', {})
            )
        }

    elif model_name.startswith('FluxNet_U_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_U_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
        }

    # FNO系列
    elif model_name.startswith('FNO_FluxD_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_FluxD_1D',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
        }

    # FNO + bound + soft (有界 + 软守恒)
    elif model_name.startswith('FNO_1D_bound_soft'):
        soft_cons = hparams.get('soft_cons_weight', 0.1)
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_1D',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                prediction_mode='residual',
                bound_mode='double',  # sigmoid有界到[0,1]
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps'],
                soft_conservation_weight=soft_cons
            )
        }

    # FNO + soft (仅软守恒)
    elif model_name.startswith('FNO_1D_soft'):
        soft_cons = hparams.get('soft_cons_weight', 0.1)
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_1D',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                prediction_mode='residual'
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps'],
                soft_conservation_weight=soft_cons
            )
        }

    elif model_name.startswith('FNO_1D'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_1D',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                prediction_mode='residual'
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
        }

    # CNN Baseline系列
    elif model_name.startswith('CNN_1D'):
        pred_mode = 'residual' if 'residual' in model_name else 'direct'
        soft_cons = 0.1 if 'soft' in model_name else 0.0
        bound_mode = 'double' if 'bound' in model_name else 'none'

        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='CNN_Baseline_1D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                prediction_mode=pred_mode,
                bound_mode=bound_mode,
                lower_bound=0.0,
                upper_bound=1.0
            ),
            'training_config': TrainingConfig(
                num_epochs=hparams['num_epochs'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                ndt=hparams['ndt'],
                num_workers=hparams['num_workers'],
                soft_conservation_weight=soft_cons,
                use_pushforward=use_pf,
                unroll_steps=hparams['unroll_steps']
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/traffic_flow/ablation_fixt"
    train_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/train"
    val_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/val"
    test_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/test_long"

    # ========================================================================
    # 超参数配置
    # ========================================================================
    hparams = {
        # 模型架构参数
        'base_channels': 32,
        'num_blocks': 6,
        'kernel_size': 5,
        'neighborhood_size': 11,
        # 训练参数
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        # FNO specific
        'fno_modes': 8,
        'fno_width': 32,
        'fno_layers': 4,
        # ================================================================
        # 损失权重配置
        # ================================================================
        # 全局损失权重模式: 'adaptive' (自适应) 或 'manual' (手动指定)
        # 'loss_weight_mode': 'adaptive',
        'loss_weight_mode': 'manual',
        # DCL损失权重 (用于FluxNet-D系列)
        'dcl_weight': 0.5,
        # 软守恒损失权重 (用于baseline模型)
        'soft_cons_weight': 0.5,
        # 手动指定损失权重 (当loss_weight_mode='manual'时生效)
        # 可用的损失项: p_loss (预测), dcl_loss (对偶一致性), stability_loss (稳定性), cons_loss (守恒)
        'loss_weights': {
            'p_loss': 0.5,
            'dcl_loss': 0.5,
            'stability_loss': 0.5,
            'cons_loss': 0.5,
        },
    }

    # ========================================================================
    # 实验控制
    # ========================================================================
    gpu_id = 3
    seed = 42
    run_training = True
    run_evaluation = True
    evaluate_mode = 'both'
    # evaluate_mode = 'rollout'
    visualize_trajectories = 'all'
    # visualize_trajectories = None

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
                dataset_type='traffic_flow',
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
