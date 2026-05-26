"""
统一训练器 - 支持所有数据集和模型

功能:
1. 支持 onestep 和 pushforward 训练 (正确实现: one-step loss + stability loss)
2. 自动适配不同模型的输出格式
3. 自适应损失权重平衡策略
4. 软约束守恒损失 (仅用于baseline模型)
5. 完整的日志和可视化
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import joblib
import numpy as np
import matplotlib.pyplot as plt

from .config import TrainingConfig


def compute_conservation_loss(pred: torch.Tensor, target: torch.Tensor,
                               normalize: bool = True) -> torch.Tensor:
    """
    软约束守恒损失 - 惩罚全局总量漂移

    Args:
        pred: 预测场 [batch, channels, ...]
        target: 目标场 [batch, channels, ...]
        normalize: 是否归一化使损失值在合理范围内

    Returns:
        守恒损失标量

    Note:
        归一化处理使守恒损失与预测MSE损失在同一数量级：
        - 原始相对误差 ~1e-3 到 1e-6（很小，导致权重过高）
        - 归一化后：转换为等效的per-element MSE scale
    """
    # 确保维度正确
    if len(pred.shape) == 2:
        # [batch, length] -> 空间维度是dim=1
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)
        num_elements = pred.shape[1]
    elif len(pred.shape) == 3:
        # [batch, channels, length] or [batch, H, W]
        if pred.shape[1] <= 4:  # likely [batch, channels, length]
            pred_sum = pred.sum(dim=2)
            target_sum = target.sum(dim=2)
            num_elements = pred.shape[2]
        else:  # likely [batch, H, W]
            pred_sum = pred.sum(dim=(1, 2))
            target_sum = target.sum(dim=(1, 2))
            num_elements = pred.shape[1] * pred.shape[2]
    else:
        # [batch, channels, H, W] or higher
        spatial_dims = list(range(2, len(pred.shape)))
        pred_sum = pred.sum(dim=spatial_dims)
        target_sum = target.sum(dim=spatial_dims)
        num_elements = 1
        for d in spatial_dims:
            num_elements *= pred.shape[d]

    # 计算总量差异
    diff = torch.abs(pred_sum - target_sum)

    if normalize:
        # 归一化方法: 转换为等效的per-element MSE scale
        # 如果总量漂移了 delta_total，平均每个元素漂移了 delta_total / N
        # MSE scale: (delta_total / N)^2 * N = delta_total^2 / N
        normalized_loss = (diff ** 2) / num_elements
        return normalized_loss.mean()
    else:
        # 原始相对误差 (用于监控，不用于训练)
        relative_diff = diff / (torch.abs(target_sum) + 1e-8)
        return relative_diff.mean()


def get_model_output_format(model) -> str:
    """
    检测模型类型，返回输出格式标识

    Returns:
        'fluxnet_d': (next, outflow, inflow) - FluxNet-D系列 (含FNO_FluxD)
        'fluxnet_sw': (next_state, h_delta, mx_delta, my_delta) - 浅水方程FluxNet
        'fluxnet_nl': (next, delta) - FluxNet-N/L/P/U系列
        'cnn': (next,) - CNN baseline
        'sw_baseline': (next,) - 浅水方程baseline (包括CNN_SW_Proj)
        'fno_sw': (next,) - FNO浅水方程 (包括FNO_SW_Proj)
        'fno': (next,) - 一般FNO
    """
    model_name = model.__class__.__name__

    if 'FluxNet_D' in model_name or 'FNO_FluxD' in model_name or 'FluxGNN_D' in model_name or 'FluxNet_D_Dirichlet' in model_name:
        return 'fluxnet_d'
    elif 'FluxNet_SW_2D' == model_name:
        return 'fluxnet_sw'
    elif 'FluxNet_SW_Baseline' == model_name:
        return 'sw_baseline'
    elif 'CNN_SW_Proj' in model_name:
        return 'sw_baseline'  # CNN_SW_Proj输出格式同sw_baseline
    elif 'FNO_SW_Proj' in model_name:
        return 'fno_sw'  # FNO_SW_Proj输出格式同fno_sw
    elif 'FNO_SW' in model_name:
        return 'fno_sw'
    elif 'FNO' in model_name or 'FluxGNN_1D' == model_name:
        return 'fno'
    elif 'FluxNet' in model_name:
        return 'fluxnet_nl'
    elif 'CNN_Baseline' in model_name:
        return 'cnn'
    else:
        raise ValueError(f"Unknown model type: {model_name}. "
                         f"Supported: FluxNet_D_*, FluxNet_SW_2D, FluxNet_SW_Baseline, "
                         f"FluxNet_N_*, FluxNet_L_*, FluxNet_P_*, FluxNet_U_*, "
                         f"CNN_Baseline_*, FNO_SW, FNO_SW_Proj, CNN_SW_Proj, FNO_1D, FNO_FluxD_1D")


def is_baseline_model(model) -> bool:
    """检查是否是baseline模型（无结构守恒保证）"""
    model_name = model.__class__.__name__
    # FNO_FluxD有守恒保证，不算baseline
    if 'FNO_FluxD' in model_name:
        return False
    if 'FluxGNN_D' in model_name or 'FluxNet_D_Dirichlet' in model_name:
        return False
    if 'FluxGNN_1D' in model_name and 'FluxGNN_D' not in model_name:
        return True  # FluxGNN_1D (without D) is a baseline-like model
    return 'Baseline' in model_name or 'CNN' in model_name or 'FNO' in model_name


class AdaptiveLossWeights:
    """
    自适应损失权重管理器

    在训练的第一个批次记录各损失项的初始值，
    以它们的几何平均值为基准计算各自的权重系数，
    使各损失项对总损失的贡献量级相近。
    """

    def __init__(self, loss_names: List[str], mode: str = 'adaptive'):
        """
        Args:
            loss_names: 损失项名称列表
            mode: 'adaptive' 自适应权重, 'manual' 用户指定权重
        """
        self.loss_names = loss_names
        self.mode = mode
        self.initial_values = {}
        self.weights = {name: 0.5 for name in loss_names}
        self.initialized = False

    def initialize(self, losses: Dict[str, torch.Tensor]):
        """首次调用时初始化权重"""
        if self.initialized or self.mode != 'adaptive':
            return

        # 记录初始值
        for name in self.loss_names:
            if name in losses:
                val = losses[name].item()
                if val > 1e-10:  # 避免除零
                    self.initial_values[name] = val

        if len(self.initial_values) == 0:
            return

        # 计算几何平均值
        values = list(self.initial_values.values())
        geo_mean = np.exp(np.mean(np.log(np.array(values) + 1e-10)))

        # 计算权重: weight_i = geo_mean / initial_i
        for name, initial in self.initial_values.items():
            self.weights[name] = geo_mean / (initial + 1e-10)

        self.initialized = True
        print(f"[AdaptiveLossWeights] Initialized weights: {self.weights}")

    def set_manual_weights(self, weight_dict: Dict[str, float]):
        """手动设置权重"""
        for name, weight in weight_dict.items():
            if name in self.weights:
                self.weights[name] = weight
        # 归一化使权重和为1
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total

    def get_weight(self, name: str) -> float:
        return self.weights.get(name, 0.5)


class UnifiedTrainer:
    """
    统一训练器

    支持所有FluxNet模型和baseline模型的训练

    Pushforward训练的正确实现:
    - 每个batch同时计算：
      1. one-step loss: 用原始输入预测下一步
      2. stability loss: 先推前N步(无梯度)，然后在第N步计算损失(有梯度)
    - total_loss = one_step_loss + stability_loss
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader,
        val_loader,
        result_dir: str,
        device: torch.device,
        dataset_type: str = 'spinodal_decomposition'
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.result_dir = result_dir
        self.device = device
        self.dataset_type = dataset_type

        # 创建输出目录
        os.makedirs(result_dir, exist_ok=True)
        self.mpdt_dir = os.path.join(result_dir, "training_visualization")
        os.makedirs(self.mpdt_dir, exist_ok=True)

        # 检测模型输出格式
        self.output_format = get_model_output_format(model)
        self.is_baseline = is_baseline_model(model)
        print(f"模型输出格式: {self.output_format}, Baseline: {self.is_baseline}")

        # 设置损失函数
        if config.loss_criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif config.loss_criterion == 'MAE':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_criterion: {config.loss_criterion}")

        # 确定损失项名称
        self.loss_names = self._get_loss_names()
        print(f"损失项: {self.loss_names}")

        # 初始化自适应权重管理器
        if config.loss_weight_mode == 'adaptive':
            self.loss_weights = AdaptiveLossWeights(self.loss_names, mode='adaptive')
        else:
            self.loss_weights = AdaptiveLossWeights(self.loss_names, mode='manual')
            if config.loss_weights:
                self.loss_weights.set_manual_weights(config.loss_weights)

        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_losses_dict = {name: [] for name in self.loss_names}
        self.val_cons_losses = []  # 守恒误差 (所有模型)
        self.best_losses = []
        self.optimizer_lrs = []

        self.best_loss = float('inf')
        self.vis_counter = 0

    def _get_loss_names(self) -> List[str]:
        """根据模型类型确定损失项名称"""
        names = ['p_loss']  # 预测损失总是存在

        # FluxNet-D的DCL损失 (仅当dcl_weight > 0时)
        if self.output_format == 'fluxnet_d' and self.config.dcl_weight > 0:
            names.append('dcl_loss')

        if self.config.use_pushforward:
            # Pushforward: 添加stability loss
            names.append('stability_loss')
            if self.output_format == 'fluxnet_d' and self.config.dcl_weight > 0:
                names.append('dcl_n_loss')  # pushforward的DCL损失
            if self.is_baseline and self.config.soft_conservation_weight > 0:
                names.append('cons_n_loss')  # pushforward的守恒损失

        if self.is_baseline and self.config.soft_conservation_weight > 0:
            names.append('cons_loss')

        return names

    def _get_model_prediction(self, model_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取模型预测和额外损失

        Returns:
            pred: 预测的下一步状态
            extra_losses: 额外的损失项 (如dcl_loss)
        """
        outputs = self.model(model_input)
        extra_losses = {}

        if self.output_format == 'fluxnet_d':
            pred, outflow, inflow = outputs
            extra_losses['dcl'] = self.criterion(outflow, inflow)
        elif self.output_format == 'fluxnet_sw':
            pred = outputs[0]
        elif self.output_format == 'fluxnet_nl':
            pred, _ = outputs
        else:  # cnn, sw_baseline, fno_sw, fno
            pred = outputs[0]

        return pred, extra_losses

    def compute_onestep_loss(self, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算one-step损失

        Args:
            inputs: 模型输入
            target: 下一步的真实值
        """
        losses = {}

        pred, extra_losses = self._get_model_prediction(inputs)

        # 主预测损失
        losses['p_loss'] = self.criterion(pred, target)

        # FluxNet-D的DCL损失 (当dcl_weight > 0时才计算)
        if 'dcl' in extra_losses and self.config.dcl_weight > 0:
            losses['dcl_loss'] = extra_losses['dcl'] * self.config.dcl_weight

        # Baseline的守恒损失 (使用归一化版本)
        if self.is_baseline and self.config.soft_conservation_weight > 0:
            cons_loss = compute_conservation_loss(pred, target, normalize=True)
            losses['cons_loss'] = cons_loss * self.config.soft_conservation_weight

        return losses, pred

    def compute_stability_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算stability/pushforward损失

        正确实现:
        1. 先推前N-1步 (完全无梯度)
        2. 在第N步做一次有梯度的前向传播
        3. 计算第N步的损失并返回

        Args:
            inputs: 初始输入 [batch, channels, ...]
            targets: 多步目标 [batch, K, ...]
        """
        losses = {}
        N = self.config.unroll_steps  # 推前步数

        # 获取当前状态
        if self.dataset_type in ['convection_diffusion', 'traffic_flow']:
            current_state = inputs[:, 0:1]  # 守恒场
            external_field = inputs[:, 1:]
            has_external = True
        elif self.dataset_type == 'shallow_water':
            current_state = inputs  # [batch, 3, H, W]
            has_external = False
        else:  # spinodal
            current_state = inputs
            has_external = False

        # ===== Step 1: 推前N-1步 (无梯度) =====
        with torch.no_grad():
            for k in range(N - 1):
                # 准备输入
                if has_external:
                    model_input = torch.cat([current_state, external_field], dim=1)
                else:
                    model_input = current_state

                # 前向传播
                pred, _ = self._get_model_prediction(model_input)
                current_state = pred

        # ===== Step 2: 第N步的有梯度前向传播 =====
        if has_external:
            model_input = torch.cat([current_state, external_field], dim=1)
        else:
            model_input = current_state

        pred, extra_losses = self._get_model_prediction(model_input)

        # 获取第N步的目标
        if self.dataset_type == 'shallow_water':
            target_N = targets[:, N-1]  # [batch, 3, H, W]
        elif self.dataset_type in ['convection_diffusion', 'traffic_flow']:
            # targets形状: [batch, K, length]
            target_N = targets[:, N-1:N]  # [batch, 1, length]
        else:  # spinodal
            # targets形状: [batch, K, H, W]
            target_N = targets[:, N-1:N]  # [batch, 1, H, W]

        # 计算stability损失
        losses['stability_loss'] = self.criterion(pred, target_N)

        # FluxNet-D的DCL损失 (当dcl_weight > 0时才计算)
        if 'dcl' in extra_losses and self.config.dcl_weight > 0:
            losses['dcl_n_loss'] = extra_losses['dcl'] * self.config.dcl_weight

        # Baseline的守恒损失 (使用归一化版本)
        if self.is_baseline and self.config.soft_conservation_weight > 0:
            cons_loss = compute_conservation_loss(pred, target_N, normalize=True)
            losses['cons_n_loss'] = cons_loss * self.config.soft_conservation_weight

        return losses

    def compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算加权总损失

        注意：某些损失项已经在计算时乘以了用户指定的权重（如cons_loss, dcl_loss），
        这些损失项不再通过AdaptiveLossWeights进行二次加权。
        """
        # 已经在计算时加权的损失项，不再进行自适应加权
        pre_weighted_losses = {'cons_loss', 'cons_n_loss', 'dcl_loss', 'dcl_n_loss'}

        # 过滤出需要自适应加权的损失
        losses_for_adaptive = {k: v for k, v in losses.items() if k not in pre_weighted_losses}

        # 初始化自适应权重 (仅对非预加权损失)
        self.loss_weights.initialize(losses_for_adaptive)

        total = 0
        for name, loss in losses.items():
            if name in pre_weighted_losses:
                # 已加权的损失直接加入
                total += loss
            else:
                # 其他损失使用自适应/手动权重
                weight = self.loss_weights.get_weight(name)
                total += weight * loss

        return total

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            all_losses = {}

            # ===== Part 1: One-step loss (始终计算) =====
            if self.config.use_pushforward:
                # pushforward模式下，one-step target是第1步
                if self.dataset_type == 'shallow_water':
                    onestep_target = targets[:, 0]  # [batch, 3, H, W]
                elif self.dataset_type in ['convection_diffusion', 'traffic_flow']:
                    onestep_target = targets[:, 0:1]  # [batch, 1, length]
                else:  # spinodal
                    onestep_target = targets[:, 0:1]  # [batch, 1, H, W]
            else:
                onestep_target = targets

            onestep_losses, _ = self.compute_onestep_loss(inputs, onestep_target)
            all_losses.update(onestep_losses)

            # ===== Part 2: Stability loss (仅pushforward模式) =====
            if self.config.use_pushforward:
                stability_losses = self.compute_stability_loss(inputs, targets)
                all_losses.update(stability_losses)

            # ===== 计算总损失并反向传播 =====
            loss = self.compute_total_loss(all_losses)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        return total_loss / num_batches

    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """验证"""
        self.model.eval()

        val_metrics = {name: 0.0 for name in self.loss_names}
        val_metrics['total_loss'] = 0.0
        val_metrics['cons_error'] = 0.0  # 统计真实守恒误差
        num_batches = 0

        vis_done = False

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                all_losses = {}

                # One-step loss
                if self.config.use_pushforward:
                    if self.dataset_type == 'shallow_water':
                        onestep_target = targets[:, 0]
                    elif self.dataset_type in ['convection_diffusion', 'traffic_flow']:
                        onestep_target = targets[:, 0:1]
                    else:
                        onestep_target = targets[:, 0:1]
                else:
                    onestep_target = targets

                onestep_losses, pred = self.compute_onestep_loss(inputs, onestep_target)
                all_losses.update(onestep_losses)

                # Stability loss
                if self.config.use_pushforward:
                    stability_losses = self.compute_stability_loss(inputs, targets)
                    all_losses.update(stability_losses)

                total_loss = self.compute_total_loss(all_losses)

                # 累积
                val_metrics['total_loss'] += total_loss.item()
                for name in self.loss_names:
                    if name in all_losses:
                        val_metrics[name] += all_losses[name].item()

                # 统计守恒误差 (使用未归一化版本用于监控)
                cons_err = compute_conservation_loss(pred, onestep_target, normalize=False).item()
                val_metrics['cons_error'] += cons_err

                num_batches += 1

                # 可视化
                if not vis_done and epoch % self.config.save_interval == 0:
                    self._save_visualization(inputs, onestep_target, pred, epoch)
                    vis_done = True

        # 平均
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)

        return val_metrics['total_loss'], val_metrics

    def _save_visualization(self, inputs, target, pred, epoch):
        """保存训练过程可视化"""
        try:
            # 根据数据集类型选择可视化
            if self.dataset_type in ['spinodal_decomposition']:
                self._visualize_2d_field(pred, target, epoch)
            elif self.dataset_type in ['convection_diffusion', 'traffic_flow']:
                self._visualize_1d_field(inputs, pred, target, epoch)
            elif self.dataset_type == 'shallow_water':
                self._visualize_shallow_water(pred, target, epoch)

        except Exception as e:
            print(f"可视化保存失败: {e}")

    def _visualize_2d_field(self, pred, target, epoch):
        """2D场可视化 (spinodal_decomposition)"""
        pred_np = pred[0, 0].cpu().numpy()
        target_np = target[0, 0].cpu().numpy()
        error_np = np.abs(pred_np - target_np)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        vmin = min(pred_np.min(), target_np.min())
        vmax = max(pred_np.max(), target_np.max())

        im0 = axes[0].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Predicted')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(target_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        im2 = axes[2].imshow(error_np, cmap='hot')
        axes[2].set_title(f'Error (MAE={error_np.mean():.4e})')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        plt.suptitle(f'Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.mpdt_dir, f'epoch_{epoch+1:04d}.png'), dpi=100)
        plt.close()

    def _visualize_1d_field(self, inputs, pred, target, epoch):
        """1D场可视化 (convection_diffusion, traffic_flow)"""
        current = inputs[0, 0].cpu().numpy()

        # 处理不同形状
        if len(pred.shape) == 3:
            pred_np = pred[0, 0].cpu().numpy()
        else:
            pred_np = pred[0].cpu().numpy()

        if len(target.shape) == 3:
            target_np = target[0, 0].cpu().numpy()
        else:
            target_np = target[0].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        x = np.arange(len(pred_np))

        # 预测 vs 真实
        axes[0].plot(x, current, 'g--', label='Current', alpha=0.7)
        axes[0].plot(x, target_np, 'b-', label='Target', linewidth=2)
        axes[0].plot(x, pred_np, 'r--', label='Predicted', linewidth=2)
        axes[0].legend()
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Prediction vs Target')
        axes[0].grid(True, alpha=0.3)

        # 误差
        error = np.abs(pred_np - target_np)
        axes[1].plot(x, error, 'g-', linewidth=2)
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title(f'Error (MAE={error.mean():.4e})')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.mpdt_dir, f'epoch_{epoch+1:04d}.png'), dpi=100)
        plt.close()

    def _visualize_shallow_water(self, pred, target, epoch):
        """浅水方程三通道可视化"""
        field_names = ['h', 'mx', 'my']

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for i, name in enumerate(field_names):
            pred_np = pred[0, i].cpu().numpy()
            target_np = target[0, i].cpu().numpy()
            error_np = np.abs(pred_np - target_np)

            vmin = min(pred_np.min(), target_np.min())
            vmax = max(pred_np.max(), target_np.max())

            im0 = axes[i, 0].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'Pred {name}')
            plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

            im1 = axes[i, 1].imshow(target_np, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'Target {name}')
            plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

            im2 = axes[i, 2].imshow(error_np, cmap='hot')
            axes[i, 2].set_title(f'Error {name}')
            plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

        plt.suptitle(f'Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.mpdt_dir, f'epoch_{epoch+1:04d}.png'), dpi=100)
        plt.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        torch.save(checkpoint, os.path.join(self.result_dir, 'latest_checkpoint.pt'))

        if is_best:
            torch.save(checkpoint, os.path.join(self.result_dir, 'best_checkpoint.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'best_model.pt'))

    def train(self) -> Dict:
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 - {self.config.num_epochs} epochs")
        print(f"训练模式: {'pushforward (one-step + stability)' if self.config.use_pushforward else 'onestep'}")
        print(f"结果保存至: {self.result_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)

            # 记录各损失项
            for name in self.loss_names:
                self.val_losses_dict[name].append(val_metrics.get(name, 0.0))

            # 记录守恒误差
            self.val_cons_losses.append(val_metrics['cons_error'])

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer_lrs.append(current_lr)

            # 保存最优模型
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.best_losses.append(self.best_loss)

            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # 打印进度
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            print(f"  Train Loss: {train_loss:.6e}")
            print(f"  Val Loss: {val_loss:.6e}, Best: {self.best_loss:.6e}")
            print(f"  Conservation Error: {val_metrics['cons_error']:.6e}")
            print(f"  LR: {current_lr:.6e}")
            print(f"  Loss weights: {self.loss_weights.weights}")

            # 保存损失曲线
            self._save_loss_curves()

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.1f}秒")
        print(f"最优验证损失: {self.best_loss:.6e}")

        self.save_checkpoint(self.config.num_epochs - 1, False)

        with open(os.path.join(self.result_dir, f"training_time_{int(total_time)}_s.txt"), 'w') as f:
            f.write(f"Total training time: {total_time:.1f} seconds\n")
            f.write(f"Best validation loss: {self.best_loss:.6e}\n")

        return {
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_time': total_time
        }

    def _save_loss_curves(self):
        """保存损失曲线"""
        # 保存数据
        loss_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_losses_dict': self.val_losses_dict,
            'val_cons_losses': self.val_cons_losses,
            'best_losses': self.best_losses,
            'optimizer_lrs': self.optimizer_lrs,
            'output_format': self.output_format,
            'is_baseline': self.is_baseline,
            'use_pushforward': self.config.use_pushforward
        }
        joblib.dump(loss_data, os.path.join(self.result_dir, "loss_curve.pkl"))

        # 绘制损失曲线
        self._plot_loss_curves(loss_data)

    def _plot_loss_curves(self, loss_data):
        """
        根据模型类型绘制合适的损失曲线

        只绘制实际用于反向传播的损失：
        - FluxNet模型：不绘制守恒损失曲线（因为守恒是架构保证的）
        - Baseline模型：绘制守恒损失（如果使用了soft conservation）
        """
        epochs = range(1, len(loss_data['train_losses']) + 1)

        # 确定需要绘制的损失项
        losses_to_plot = []
        losses_to_plot.append(('Train Loss', loss_data['train_losses'], 'blue'))
        losses_to_plot.append(('Val Loss', loss_data['val_losses'], 'orange'))

        # 根据模型类型添加相应的损失
        # DCL loss (dual consistency loss, 原io_loss)
        if self.output_format == 'fluxnet_d' and 'dcl_loss' in loss_data['val_losses_dict']:
            losses_to_plot.append(('DCL Loss', loss_data['val_losses_dict']['dcl_loss'], 'green'))

        if 'p_loss' in loss_data['val_losses_dict']:
            losses_to_plot.append(('Pred Loss', loss_data['val_losses_dict']['p_loss'], 'red'))

        # 只有baseline模型才绘制守恒误差曲线 (FluxNet守恒是架构保证的，无需绘制)
        if self.is_baseline:
            losses_to_plot.append(('Conservation Error', loss_data['val_cons_losses'], 'purple'))

        # Pushforward添加stability loss
        if self.config.use_pushforward and 'stability_loss' in loss_data['val_losses_dict']:
            losses_to_plot.append(('Stability Loss', loss_data['val_losses_dict']['stability_loss'], 'cyan'))

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 主损失图
        for name, data, color in losses_to_plot[:4]:  # 前4个放在主图
            ax1.plot(epochs, data, label=name, color=color if isinstance(color, str) else None, linewidth=2)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Training Curves - {self.output_format}')

        # 学习率
        ax1_lr = ax1.twinx()
        ax1_lr.plot(epochs, loss_data['optimizer_lrs'], 'k--', alpha=0.5, label='LR')
        ax1_lr.set_ylabel('Learning Rate', color='gray')
        ax1_lr.tick_params(axis='y', labelcolor='gray')

        # 次要损失图 (如果有更多损失)
        if len(losses_to_plot) > 4:
            for name, data, color in losses_to_plot[4:]:
                ax2.plot(epochs, data, label=name, linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Additional Loss Components')
        else:
            ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'loss_curve.png'), dpi=150)
        plt.close()


def train_model(
    model: nn.Module,
    dataset_type: str,
    train_folder: str,
    val_folder: str,
    result_dir: str,
    config: TrainingConfig,
    device: torch.device = None,
    num_workers: int = 4
) -> Dict:
    """
    统一训练接口
    """
    from .dataloader import create_data_loader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_mode = 'pushforward' if config.use_pushforward else 'onestep'

    train_loader = create_data_loader(
        dataset_type=dataset_type,
        folder_path=train_folder,
        batch_size=config.batch_size,
        ndt=config.ndt,
        shuffle=True,
        num_workers=num_workers,
        training_mode=training_mode,
        unroll_steps=config.unroll_steps
    )

    val_loader = create_data_loader(
        dataset_type=dataset_type,
        folder_path=val_folder,
        batch_size=config.batch_size,
        ndt=config.ndt,
        shuffle=False,
        num_workers=num_workers,
        training_mode=training_mode,
        unroll_steps=config.unroll_steps
    )

    trainer = UnifiedTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        result_dir=result_dir,
        device=device,
        dataset_type=dataset_type
    )

    return trainer.train()
