"""
浅水方程 - 消融实验

可选模型完整列表（从中选取需要的进行消融实验）：

FluxNet模型系列:
- FluxNet_SW_LAP: L-head for h + Advection-Pressure decomposition (RECOMMENDED)
- FluxNet_SW_PAP: P-head for h + Advection-Pressure decomposition (ablation)
- FluxNet_SW_LAP_no_gate: LAP without h^2 pressure gate (ablation)
- FluxNet_SW_PPP: P-head for all fields (ablation)
- FluxNet_SW_LPP: L-head for h, P-head for mx/my (ablation)
- FluxNet_SW_NNN: No constraint (deprecated)

Baseline模型系列:
- SW_Baseline_direct: 直接预测
- SW_Baseline_residual: 残差预测
- SW_Baseline_bound: 带h下界约束
- SW_Baseline_soft: 带软守恒损失

其他模型:
- FNO_SW: FNO基线

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
    # FluxNet系列 - onestep (推荐: LAP)
    'FluxNet_SW_LAP_onestep',       # RECOMMENDED: L-head + Advection-Pressure
    'FluxNet_SW_PAP_onestep',       # ablation: P-head + Advection-Pressure
    'FluxNet_SW_LAP_no_gate_onestep',  # ablation: LAP without h^2 gate
    'FluxNet_SW_PPP_onestep',       # ablation: all P-head
    'FluxNet_SW_LPP_onestep',       # ablation: L-head for h, P-head for m
    'FluxNet_SW_NNN_onestep',       # deprecated: no constraint
    # FluxNet系列 - pushforward
    'FluxNet_SW_LAP_pf',
    'FluxNet_SW_PAP_pf',
    'FluxNet_SW_LAP_no_gate_pf',
    'FluxNet_SW_PPP_pf',
    'FluxNet_SW_LPP_pf',
    'FluxNet_SW_NNN_pf',
    # Baseline系列 (old CNN direct) - onestep
    'SW_Baseline_direct_onestep',
    'SW_Baseline_residual_onestep',
    'SW_Baseline_direct_soft_onestep',
    'SW_Baseline_residual_soft_onestep',
    'SW_Baseline_bound_onestep',
    'SW_Baseline_bound_soft_onestep',
    # Baseline系列 (old CNN direct) - pushforward
    'SW_Baseline_direct_pf',
    'SW_Baseline_residual_pf',
    'SW_Baseline_direct_soft_pf',
    'SW_Baseline_residual_soft_pf',
    'SW_Baseline_bound_pf',
    'SW_Baseline_bound_soft_pf',
    # FNO (no projection)
    'FNO_SW_onestep',
    'FNO_SW_pf',
    # FNO + soft conservation loss
    'FNO_SW_soft_onestep',          # FNO + soft conservation loss
    'FNO_SW_soft_pf',
    # FNO + Projection (strong baseline)
    'FNO_SW_Proj_box_onestep',      # FNO + box projection (h>=0)
    'FNO_SW_Proj_box_mass_onestep', # FNO + box + mass projection (RECOMMENDED baseline)
    'FNO_SW_Proj_box_pf',
    'FNO_SW_Proj_box_mass_pf',
    # CNN + Projection (strong baseline)
    'CNN_SW_Proj_box_onestep',      # CNN + box projection
    'CNN_SW_Proj_box_mass_onestep', # CNN + box + mass projection (RECOMMENDED baseline)
    'CNN_SW_Proj_box_pf',
    'CNN_SW_Proj_box_mass_pf',
    # FNO + FluxLAP head (ablation: FNO backbone with LAP head)
    'FNO_FluxLAP_onestep',
    'FNO_FluxLAP_pf',
]


# ============================================================================
SELECTED_MODELS = [
    # ===== 我的方法 (FluxNet-LAP) =====
    'FluxNet_SW_LAP_pf',              # 最优方法: L-head + Adv-Pressure + pushforward

    # ===== 基线对比 =====
    'FNO_SW_pf',                      # FNO基线
    'FNO_SW_soft_pf',                 # FNO + 软守恒损失
    'SW_Baseline_residual_pf',        # CNN基线 (残差预测)
    'SW_Baseline_residual_soft_pf',   # CNN + 软守恒损失
    'FNO_SW_Proj_box_mass_pf',        # FNO + Box + Mass投影 (强基线)
    'CNN_SW_Proj_box_mass_pf',        # CNN + Box + Mass投影 (强基线)

    # ===== 消融实验 =====
    'FluxNet_SW_PPP_pf',              # 消融: 三场都用P头
    'FluxNet_SW_LPP_pf',              # 消融: h用L头, m用P头
    'FluxNet_SW_PAP_pf',              # 消融: h用P头 + Adv-Pressure
    'FluxNet_SW_LAP_no_gate_pf',      # 消融: LAP无press_scale门控
    'FluxNet_SW_LAP_onestep',         # 消融: LAP无pushforward
    'FNO_FluxLAP_pf',                 # 消融: FNO backbone + LAP头
]


def _make_training_config(hparams: dict, use_pf: bool, soft_cons: float = 0.0) -> TrainingConfig:
    """
    创建TrainingConfig的辅助函数，统一处理loss_weight_mode

    Args:
        hparams: 超参数字典
        use_pf: 是否使用pushforward训练
        soft_cons: 软守恒损失权重 (仅baseline使用)

    Returns:
        TrainingConfig实例

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

    Args:
        model_name: 模型名称
        hparams: 超参数字典

    Returns:
        实验配置字典

    损失权重使用说明:
        - 默认使用自适应损失权重 (loss_weight_mode='adaptive')
        - 手动指定时，设置 hparams['loss_weight_mode'] = 'manual'
        - 然后在 hparams['loss_weights'] 中指定各损失权重
        - 可用损失项: p_loss, stability_loss, cons_loss
    """
    use_pf = '_pf' in model_name

    # FluxNet_SW系列 - 新的head配置
    # LAP: RECOMMENDED - L-head + Advection-Pressure decomposition
    if model_name.startswith('FluxNet_SW_LAP_no_gate'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='LAP_no_gate',
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_SW_LAP'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='LAP',
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_SW_PAP'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='PAP',
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_SW_PPP'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='PPP',
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_SW_LPP'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='LPP',
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    elif model_name.startswith('FluxNet_SW_NNN'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_2D',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                neighborhood_size=hparams['neighborhood_size'],
                head_config='NNN'
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    # Baseline系列
    elif model_name.startswith('SW_Baseline'):
        pred_mode = 'residual' if 'residual' in model_name else 'direct'
        soft_cons = hparams.get('soft_cons_weight', 0.1) if 'soft' in model_name else 0.0
        bound_h = 'bound' in model_name

        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FluxNet_SW_Baseline',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                prediction_mode=pred_mode,
                bound_h=bound_h,
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf, soft_cons)
        }

    # FNO + soft conservation loss
    elif model_name.startswith('FNO_SW_soft'):
        soft_cons = hparams.get('soft_cons_weight', 0.1)
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_SW',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4)
            ),
            'training_config': _make_training_config(hparams, use_pf, soft_cons)
        }

    # FNO系列 (no projection)
    elif model_name.startswith('FNO_SW') and 'Proj' not in model_name and 'FluxLAP' not in model_name:
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_SW',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4)
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    # FNO + Projection
    elif model_name.startswith('FNO_SW_Proj'):
        proj_mode = 'box_mass' if 'box_mass' in model_name else 'box'
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_SW_Proj',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                projection_mode=proj_mode,
                prediction_mode='residual'
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    # CNN + Projection
    elif model_name.startswith('CNN_SW_Proj'):
        proj_mode = 'box_mass' if 'box_mass' in model_name else 'box'
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='CNN_SW_Proj',
                base_channels=hparams['base_channels'],
                num_blocks=hparams['num_blocks'],
                kernel_size=hparams['kernel_size'],
                projection_mode=proj_mode,
                prediction_mode='residual'
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    # FNO + FluxLAP head (ablation: FNO backbone with LAP conservation head)
    elif model_name.startswith('FNO_FluxLAP'):
        return {
            'name': model_name,
            'model_config': ModelConfig(
                model_type='FNO_FluxLAP',
                modes=hparams.get('fno_modes', 16),
                width=hparams.get('fno_width', 64),
                num_layers=hparams.get('fno_layers', 4),
                neighborhood_size=hparams['neighborhood_size'],
                lower_bound=0.0
            ),
            'training_config': _make_training_config(hparams, use_pf)
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/shallow_water/ablation_unroll"
    train_folder = "FluxNet/dataset/shallow_water/train"
    val_folder = "FluxNet/dataset/shallow_water/val"
    test_folder = "FluxNet/dataset/shallow_water/test_long"

    # ========================================================================
    # 超参数配置
    # ========================================================================
    hparams = {
        # 模型架构参数
        'base_channels': 64,
        'num_blocks': 6,
        'kernel_size': 5,
        'neighborhood_size': 3,
        # 训练参数
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        # FNO specific
        'fno_modes': 16,
        'fno_width': 64,
        'fno_layers': 4,
        # ================================================================
        # 损失权重配置
        # ================================================================
        # 'loss_weight_mode': 'adaptive',
        'loss_weight_mode': 'manual',
        'soft_cons_weight': 0.5,
        'loss_weights': {
            'p_loss': 0.5,
            'stability_loss': 0.5,
            'cons_loss': 0.5,
        },
    }

    # ========================================================================
    # 实验控制
    # ========================================================================
    gpu_id = 0
    seed = 42
    run_training = True
    run_evaluation = True
    evaluate_mode = 'both'        # 'onestep', 'rollout', 'both'
    # evaluate_mode = 'rollout'        # 'onestep', 'rollout', 'both'
    visualize_trajectories = None  # 'all', None, or list of h5 paths
    # visualize_trajectories = 'all'  # 'all', None, or list of h5 paths
    # visualize_trajectories = None  # 'all', None, or list of h5 paths

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
                dataset_type='shallow_water',
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
