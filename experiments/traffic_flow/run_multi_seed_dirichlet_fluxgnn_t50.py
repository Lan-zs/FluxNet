"""
交通流 (Dirichlet边界) - FluxGNN_1D (t50) 多随机种子实验

基于 run_ablation_boundary_fluxgnn_t50.py 的配置，使用20个随机种子
结果目录结构: save_path/RandomSeed{seed}/FluxGNN_1D_pf/
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.common.experiment_runner import ModelConfig, TrainingConfig
from experiments.common.multi_seed_runner import (
    run_multi_seed_experiment, generate_seeds, METRICS_1D_WITH_FINAL
)


SELECTED_MODELS = ['FluxGNN_1D_pf']


def get_experiment_config(model_name: str, hparams: dict) -> dict:
    """生成实验配置"""
    return {
        'name': model_name,
        'model_config': ModelConfig(
            model_type='FluxGNN_1D',
            modes=hparams['fno_modes'],
            width=hparams['fno_width'],
            num_layers=hparams['fno_layers'],
            prediction_mode='residual',
        ),
        'training_config': TrainingConfig(
            num_epochs=hparams['num_epochs'],
            batch_size=hparams['batch_size'],
            learning_rate=hparams['learning_rate'],
            weight_decay=hparams['weight_decay'],
            ndt=hparams['ndt'],
            num_workers=hparams['num_workers'],
            use_pushforward=True,
            unroll_steps=hparams['unroll_steps'],
            loss_weight_mode=hparams['loss_weight_mode'],
            loss_weights=hparams['loss_weights'],
        ),
    }


def main():
    # ========================================================================
    # 路径配置 (s1t50数据集)
    # ========================================================================
    save_path = "FluxNet/results/traffic_flow/multi_seed_boundary_fluxgnn_s1t50"
    train_folder = "FluxNet/dataset/traffic_flow_fix_t_space1_boundary_s1t50/train"
    val_folder = "FluxNet/dataset/traffic_flow_fix_t_space1_boundary_s1t50/val"
    test_folder = "FluxNet/dataset/traffic_flow_fix_t_space1_boundary_s1t50/test_long"

    # ========================================================================
    # 超参数配置
    # ========================================================================
    hparams = {
        'fno_modes': 8,
        'fno_width': 32,
        'fno_layers': 4,
        'num_epochs': 300,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'ndt': 1,
        'num_workers': 4,
        'unroll_steps': 5,
        'loss_weight_mode': 'manual',
        'loss_weights': {
            'p_loss': 0.5,
            'stability_loss': 0.5,
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