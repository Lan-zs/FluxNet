"""
交通流 - 多随机种子实验

基于 run_single_seed_periodic.py 的配置，使用20个随机种子进行训练+评估
结果目录结构: save_path/RandomSeed{seed}/{model_name}/
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.common.multi_seed_runner import (
    run_multi_seed_experiment, generate_seeds, METRICS_1D_WITH_FINAL
)
from experiments.traffic_flow.run_single_seed_periodic import (
    get_experiment_config
)

SELECTED_MODELS = [
    # ===== 我的方法 (FluxNet-D) =====
    'FluxNet_D_1D_pf',              # 最优方法: 双界约束 + pushforward

    # ===== 基线对比 =====
    'FNO_1D_pf',                    # FNO基线
    'FNO_1D_soft_pf',               # FNO + 软守恒损失
    'CNN_1D_residual_pf',           # CNN基线 (残差预测)
    'CNN_1D_residual_soft_pf',      # CNN + 软守恒损失
    'CNN_1D_bound_soft_pf',         # CNN + 有界(sigmoid) + 软守恒损失
    'FNO_FluxD_1D_pf',              # 消融: FNO backbone + D头

    # ===== 消融实验 =====
    'FluxNet_P_1D_pf',              # 消融: P头 (正通量)
    'FluxNet_L_1D_pf',              # 消融: L头 (仅下界)
    'FluxNet_U_1D_pf',              # 消融: U头 (仅上界)
    'FluxNet_D_1D_no_dcl_pf',       # 消融: D头 无对偶一致性损失
    'FluxNet_D_1D_onestep',         # 消融: D头 无pushforward

]


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/traffic_flow/multi_seed_fixt"
    train_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/train"
    val_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/val"
    test_folder = "FluxNet/dataset/traffic_flow_fix_t_space1/test_long"

    # ========================================================================
    # 超参数配置 (与run_ablation.py一致)
    # ========================================================================
    hparams = {
        'base_channels': 32,
        'num_blocks': 6,
        'kernel_size': 5,
        'neighborhood_size': 11,
        'num_epochs': 300,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        'fno_modes': 8,
        'fno_width': 32,
        'fno_layers': 4,
        'loss_weight_mode': 'manual',
        'dcl_weight': 0.5,
        'soft_cons_weight': 0.5,
        'loss_weights': {
            'p_loss': 0.5,
            'dcl_loss': 0.5,
            'stability_loss': 0.5,
            'cons_loss': 0.5,
        },
    }

    # ========================================================================
    # 多种子实验配置
    # ========================================================================
    num_seeds = 5
    seeds = generate_seeds(num_seeds, base_seed=42)
    gpu_id = 0
    evaluate_mode = 'rollout'
    visualize_trajectories = None

    print(f"随机种子列表: {seeds}")

    # ========================================================================
    # 运行多种子实验
    # ========================================================================
    run_multi_seed_experiment(
        save_path=save_path,
        seeds=seeds,
        get_experiment_config_fn=get_experiment_config,
        selected_models=SELECTED_MODELS,
        hparams=hparams,
        dataset_type='traffic_flow',
        train_folder=train_folder,
        val_folder=val_folder,
        test_folder=test_folder,
        gpu_id=gpu_id,
        evaluate_mode=evaluate_mode,
        visualize_trajectories=visualize_trajectories,
        metrics_keys=METRICS_1D_WITH_FINAL,
    )


if __name__ == "__main__":
    main()