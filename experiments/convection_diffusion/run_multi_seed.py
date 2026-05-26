"""
对流扩散 - 多随机种子实验

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
from experiments.convection_diffusion.run_single_seed import (
    get_experiment_config, SELECTED_MODELS
)


def main():
    # ========================================================================
    # 路径配置 (使用绝对路径)
    # ========================================================================
    save_path = "FluxNet/results/convection_diffusion/multi_seed_onestep"
    train_folder = "FluxNet/dataset/convection_diffusion/train"
    val_folder = "FluxNet/dataset/convection_diffusion/val"
    test_folder = "FluxNet/dataset/convection_diffusion/test"

    # ========================================================================
    # 超参数配置 (与run_ablation.py一致)
    # ========================================================================
    hparams = {
        'base_channels': 16,
        'num_blocks': 4,
        'kernel_size': 3,
        'neighborhood_size': 3,
        'num_epochs': 300,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        'loss_weight_mode': 'adaptive',
        'soft_cons_weight': 0.1,
        'loss_weights': {
            'p_loss': 1.0,
            'stability_loss': 0.5,
            'cons_loss': 0.1,
        },
    }

    # ========================================================================
    # 多种子实验配置
    # ========================================================================
    num_seeds = 5
    seeds = generate_seeds(num_seeds, base_seed=42)
    gpu_id = 1
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
        dataset_type='convection_diffusion',
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