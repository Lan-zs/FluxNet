import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad(nn.Module):
    def __init__(self, padding):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, norm_2d=nn.BatchNorm2d):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            # 使用周期性填充
            CircularPad(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn(),
            CircularPad(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)

class FluxNet_L(nn.Module):
    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 neighborhood_size=15
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size

        # Total number of channels: 1 for outflow percentage + (neighborhood_size^2 - 1) for distribution
        self.total_channels = 1 + (neighborhood_size * neighborhood_size - 1)
        self.num_neighbors = neighborhood_size * neighborhood_size - 1

        # First convolution layer
        self.first_conv = nn.Sequential(
            CircularPad(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(nn.ModuleList([
                DoubleConv(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Flux prediction layer (1x1 conv mapping from base_channels to total_channels)
        # First channel: outflow percentage, Remaining channels: distribution ratios
        self.flux_conv = nn.Conv2d(base_channels, self.total_channels, kernel_size=1)

        # Generate the neighbor offsets for the 15x15 neighborhood
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i != 0 or j != 0:  # Exclude center point
                    neighbor_offsets.append((i, j))
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    def forward(self, x):
        # Initial feature extraction
        features = self.first_conv(x)

        # Process through residual blocks
        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # Predict fluxes to all neighbors
        raw_fluxes = self.flux_conv(features)  # [batch, total_channels, height, width]

        # Split the raw fluxes
        outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])  # First channel -> sigmoid for outflow percentage
        distribution_logits = raw_fluxes[:, 1:]  # Remaining channels -> softmax for distribution

        # Apply softmax to get distribution ratios across neighbors
        distribution_ratios = F.softmax(distribution_logits, dim=1)

        # Compute next solute field using the predicted outflow and distribution
        solute_field = x[:, 0:1]  # Assuming first channel is the solute concentration
        next_field = self._compute_transport(solute_field, outflow_percentage, distribution_ratios)

        return next_field, next_field-solute_field

    def _compute_transport(self, current_field, outflow_percentage, distribution_ratios):
        """
        Vectorized implementation of solute transport computation based on
        outflow percentage and distribution ratios.
        """
        # Create a new field to store the result
        next_field = current_field.clone()

        # Calculate total outflow amount for each cell
        outflow_amount = current_field * outflow_percentage  # [batch, 1, height, width]

        # Subtract the outflow from the current field
        next_field = next_field - outflow_amount

        # Calculate the absolute amount distributed to each neighbor
        neighbor_flows = outflow_amount * distribution_ratios  # [batch, num_neighbors, height, width]

        # For each neighbor, shift the flow tensor and add to the destination positions
        for n, (dh, dw) in enumerate(self.neighbor_offsets):
            # Extract the flow to this specific neighbor direction
            neighbor_flow = neighbor_flows[:, n:n+1]  # [batch, 1, height, width]

            # Shift the flow tensor in the opposite direction of the offset
            # This aligns the flow with the destination cell
            shifted_flow = self._shift_tensor(neighbor_flow, -dh, -dw)

            # Add the incoming flow to the destination cells
            next_field = next_field + shifted_flow

        return next_field

    def _shift_tensor(self, x, dh, dw):
        """
        Shift tensor with periodic boundary conditions
        """
        # Roll the tensor along both dimensions
        return torch.roll(x, shifts=(dh, dw), dims=(2, 3))


import torch
import torch.nn as nn
import time
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 使用示例和测试
if __name__ == "__main__":
    setup_seed(1234)

    # 测试不同的输入通道数和尺寸组合
    test_configs = [
        {'in_channels': 1, 'sizes': [(2, 1, 24, 24)]},
    ]

    for config in test_configs:
        print(f"\n=== Testing model with {config['in_channels']} input channels ===")
        # 创建模型实例
        model = FluxNet_L(
            in_channels=config['in_channels'],
            base_channels=64,
            num_blocks=4,
            kernel_size=3,
            act_fn=nn.GELU,
            norm_2d=nn.BatchNorm2d,
            neighborhood_size=15
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        model.to(device)

        for size in config['sizes']:
            print(f"\nInput size: {size}")

            # 创建一个随机但有意义的溶质场（值在0到1之间的浓度场）
            solute_field = torch.rand(size).to(device)

            # 计算初始总溶质量
            initial_total_mass = solute_field.sum().item()
            print(f"Initial total solute mass: {initial_total_mass:.6f}")

            # 记录每个模块的执行时间
            timing_results = {}

            # 1. 测量特征提取过程时间
            start_time = time.time()
            features = model.first_conv(solute_field)
            torch.cuda.synchronize()
            timing_results['feature_extraction'] = time.time() - start_time

            # 2. 测量残差块处理时间
            start_time = time.time()
            for main_path, fusion_conv in model.res_blocks:
                identity = features
                features = main_path(features)
                features = torch.cat([features, identity], dim=1)
                features = fusion_conv(features)
            torch.cuda.synchronize()
            timing_results['residual_blocks'] = time.time() - start_time

            # 3. 测量通量预测时间
            start_time = time.time()
            raw_fluxes = model.flux_conv(features)

            # 分离并处理两种不同类型的通量
            outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])
            distribution_logits = raw_fluxes[:, 1:]
            distribution_ratios = F.softmax(distribution_logits, dim=1)

            torch.cuda.synchronize()
            timing_results['flux_prediction'] = time.time() - start_time

            # 4. 测量运输计算时间
            start_time = time.time()
            next_field = model._compute_transport(solute_field, outflow_percentage, distribution_ratios)
            torch.cuda.synchronize()
            timing_results['transport_computation'] = time.time() - start_time

            # 5. 整体前向传播时间
            start_time = time.time()
            next_field_full, _ = model(solute_field)
            torch.cuda.synchronize()
            timing_results['full_forward'] = time.time() - start_time

            # 验证守恒性
            final_total_mass = next_field_full.sum().item()
            mass_difference = final_total_mass - initial_total_mass
            print(f"Final total solute mass: {final_total_mass:.6f}")
            print(f"Mass difference: {mass_difference:.8f} ({(mass_difference / initial_total_mass) * 100:.8f}%)")

            # 输出各模块执行时间
            print("\nPerformance Breakdown:")
            print(f"{'Module':<25} {'Time (ms)':<12} {'Percentage':<10}")
            print("-" * 50)
            full_time = timing_results['full_forward'] * 1000  # 转换为毫秒
            for module, t in timing_results.items():
                ms_time = t * 1000  # 转换为毫秒
                percentage = (ms_time / full_time) * 100
                print(f"{module:<25} {ms_time:<12.2f} {percentage:<10.2f}%")

            # 测试多步迭代的守恒性
            print("\n=== Testing multi-step conservation ===")
            current_field = solute_field.clone()
            num_steps = 10
            step_masses = []

            for step in range(num_steps):
                with torch.no_grad():
                    current_field, _ = model(current_field)
                    step_mass = current_field.sum().item()
                    step_masses.append(step_mass)
                    mass_diff_pct = (step_mass - initial_total_mass) / initial_total_mass * 100
                    print(f"Step {step + 1}: Total mass = {step_mass:.6f}, Diff = {mass_diff_pct:.8f}%")

            # 计算多步迭代的最大质量变化率
            max_diff_pct = max([abs((m - initial_total_mass) / initial_total_mass * 100) for m in step_masses])
            print(f"Maximum mass difference over {num_steps} steps: {max_diff_pct:.8f}%")
