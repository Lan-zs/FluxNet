"""
浅水方程 - 多随机种子实验

基于 run_single_seed_periodic.py 的配置，使用20个随机种子进行训练+评估
结果目录结构: save_path/RandomSeed{seed}/{model_name}/
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.common.multi_seed_runner import (
    run_multi_seed_experiment, generate_seeds, METRICS_SW
)
from experiments.shallow_water.run_single_seed import (
    get_experiment_config
)

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


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/shallow_water/multi_seed_unroll"
    train_folder = "FluxNet/dataset/shallow_water/train"
    val_folder = "FluxNet/dataset/shallow_water/val"
    test_folder = "FluxNet/dataset/shallow_water/test_long"

    # ========================================================================
    # 超参数配置 (与run_ablation.py一致)
    # ========================================================================
    hparams = {
        'base_channels': 64,
        'num_blocks': 6,
        'kernel_size': 5,
        'neighborhood_size': 3,
        'num_epochs': 300,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        'fno_modes': 16,
        'fno_width': 64,
        'fno_layers': 4,
        'loss_weight_mode': 'manual',
        'soft_cons_weight': 0.5,
        'loss_weights': {
            'p_loss': 0.5,
            'stability_loss': 0.5,
            'cons_loss': 0.5,
        },
    }

    # ========================================================================
    # 多种子实验配置
    # ========================================================================
    num_seeds = 5
    seeds = generate_seeds(num_seeds, base_seed=42)
    gpu_id = 2
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
        dataset_type='shallow_water',
        train_folder=train_folder,
        val_folder=val_folder,
        test_folder=test_folder,
        gpu_id=gpu_id,
        evaluate_mode=evaluate_mode,
        visualize_trajectories=visualize_trajectories,
        metrics_keys=METRICS_SW,
    )


if __name__ == "__main__":
    main()