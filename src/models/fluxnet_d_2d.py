import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random


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


class FluxNet_D(nn.Module):
    """
    Dual-Bounded Flux Network for conservative and bounded solute transport prediction.

    This network ensures:
    1. Mass conservation through flux-based prediction
    2. Lower and upper bounds through dual outflow/inflow approaches
    3. Learnable or fixed boundary parameters
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.GELU,
                 norm_2d=nn.BatchNorm2d,
                 neighborhood_size=15,
                 lower_bound=0.0,
                 upper_bound=1.0,
                 learnable_lower_bound=False,
                 learnable_upper_bound=False
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size
        self.learnable_lower_bound = learnable_lower_bound
        self.learnable_upper_bound = learnable_upper_bound

        if learnable_lower_bound:
            # 1. 计算 logit 值 (结果是一个 Tensor)
            logit_tensor = self._inverse_sigmoid(lower_bound)
            # 2. 使用 .data 来初始化 nn.Parameter，避免警告
            # .data 提供了底层数据，安全地创建了新的 Parameter
            self.lower_bound_logit = nn.Parameter(logit_tensor.data)
        else:
            # 1. 确保 lower_bound 是一个 Tensor
            bound_tensor = torch.as_tensor(lower_bound)
            # 2. 注册 Buffer，使用 .detach() 确保其不跟踪梯度
            self.register_buffer('lower_bound_value', bound_tensor.detach())

        if learnable_upper_bound:
            # 1. 计算 logit 值
            logit_tensor_upper = self._inverse_sigmoid(upper_bound)
            # 2. 使用 .data 初始化 nn.Parameter
            self.upper_bound_logit = nn.Parameter(logit_tensor_upper.data)
        else:
            # 1. 确保 upper_bound 是一个 Tensor
            bound_tensor_upper = torch.as_tensor(upper_bound)
            # 2. 注册 Buffer，使用 .detach()
            self.register_buffer('upper_bound_value', bound_tensor_upper.detach())

        # Number of neighbors (excluding center point itself)
        self.num_neighbors = neighborhood_size * neighborhood_size - 1

        # Total channels needed for dual approach: 2 sets of [1 outflow percentage + num_neighbors distribution ratios]
        self.total_channels = 2 * (1 + self.num_neighbors)

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

        # Flux prediction layer for both outflow and inflow approaches
        self.flux_conv = nn.Conv2d(base_channels, self.total_channels, kernel_size=1)

        # Generate the neighbor offsets for the neighborhood
        radius = neighborhood_size // 2
        neighbor_offsets = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i != 0 or j != 0:  # Exclude center point
                    neighbor_offsets.append((i, j))
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.long))

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-7):
        """Inverse sigmoid (logit) function for parameter initialization"""
        x = torch.clamp(torch.tensor(x), eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def lower_bound(self):
        """Get current lower bound value"""
        if self.learnable_lower_bound:
            return torch.sigmoid(self.lower_bound_logit)
        else:
            return self.lower_bound_value

    @property
    def upper_bound(self):
        """Get current upper bound value"""
        if self.learnable_upper_bound:
            return torch.sigmoid(self.upper_bound_logit)
        else:
            return self.upper_bound_value

    def forward(self, x):
        # Initial feature extraction
        features = self.first_conv(x)

        # Process through residual blocks
        for main_path, fusion_conv in self.res_blocks:
            identity = features
            features = main_path(features)
            features = torch.cat([features, identity], dim=1)
            features = fusion_conv(features)

        # Predict fluxes for both approaches
        raw_fluxes = self.flux_conv(features)  # [batch, total_channels, height, width]

        # Split the raw fluxes for outflow approach (lower bound)
        outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])
        outflow_distribution_logits = raw_fluxes[:, 1:self.num_neighbors + 1]
        outflow_distribution_ratios = F.softmax(outflow_distribution_logits, dim=1)

        # Split the raw fluxes for inflow approach (upper bound)
        inflow_percentage = torch.sigmoid(raw_fluxes[:, self.num_neighbors + 1:self.num_neighbors + 2])
        inflow_distribution_logits = raw_fluxes[:, self.num_neighbors + 2:]
        inflow_distribution_ratios = F.softmax(inflow_distribution_logits, dim=1)

        # Compute next solute field using both approaches
        solute_field = x[:, 0:1]  # Assuming first channel is the solute concentration

        # Get current bounds
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Compute changes from both approaches
        outflow_change, inflow_change = self._compute_transport(
            solute_field,
            outflow_percentage,
            outflow_distribution_ratios,
            inflow_percentage,
            inflow_distribution_ratios,
            lower_bound,
            upper_bound
        )

        # Average the changes from both approaches
        combined_change = (outflow_change + inflow_change) / 2

        # Apply the combined change to the input field
        next_field = solute_field + combined_change

        return next_field, outflow_change, inflow_change


    def _compute_transport(self, current_field, outflow_percentage, outflow_distribution_ratios,
                                    inflow_percentage, inflow_distribution_ratios, lower_bound, upper_bound):
        """
        Alternative highly optimized version using unfold/fold operations.

        This approach uses im2col-style operations for maximum parallelism.
        Note: This is more memory intensive but potentially faster for large neighborhoods.
        """
        batch_size, _, height, width = current_field.shape
        radius = self.neighborhood_size // 2

        # ------ Outflow approach ------
        available_for_outflow = current_field - lower_bound
        outflow_amount = available_for_outflow * outflow_percentage
        outflow_change = -outflow_amount
        # Pre-compute all flows
        outflow_to_all = outflow_amount * outflow_distribution_ratios  # [B, num_neighbors, H, W]

        available_for_inflow = upper_bound - current_field
        inflow_amount = available_for_inflow * inflow_percentage
        inflow_change = inflow_amount
        inflow_from_all = inflow_amount * inflow_distribution_ratios  # [B, num_neighbors, H, W]

        # Vectorized shift and accumulate
        for n, (dh, dw) in enumerate(self.neighbor_offsets):
            # Outflow
            shifted_out = torch.roll(outflow_to_all[:, n:n + 1], shifts=(-dh, -dw), dims=(2, 3))
            outflow_change = outflow_change + shifted_out

            # Inflow
            shifted_in = torch.roll(inflow_from_all[:, n:n + 1], shifts=(dh, dw), dims=(2, 3))
            inflow_change = inflow_change - shifted_in

        return outflow_change, inflow_change



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 使用示例和测试
if __name__ == "__main__":
    setup_seed(123)

    # 测试不同的输入通道数和尺寸组合
    test_configs = [
        {'in_channels': 1, 'sizes': [(2, 1, 24, 24)]},
    ]

    for config in test_configs:
        print(f"\n=== Testing model with {config['in_channels']} input channels ===")
        # 创建模型实例（测试可学习边界）
        model = FluxNet_D(
            in_channels=config['in_channels'],
            base_channels=32,
            num_blocks=4,
            kernel_size=3,
            act_fn=nn.GELU,
            norm_2d=nn.BatchNorm2d,
            neighborhood_size=3,
            lower_bound=0,
            upper_bound=1,
            learnable_lower_bound=False,  # 测试可学习下界
            learnable_upper_bound=False  # 测试可学习上界
        )


        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        model.to(device)

        model.eval()

        # print("正在编译模型 (第一次运行会稍慢)...")
        # optimized_model = torch.compile(model, mode="reduce-overhead")

        # 4. 创建假数据测试
        input_tensor = torch.rand(10, 1, 512, 512).to(device)

        # 5. 预热 (Warmup) - 编译发生在第一次运行时
        with torch.no_grad():
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            print(f"Warmup (Compilation) time: {time.time() - start:.4f}s")

        # 6. 正式测速
        print("开始测速...")
        start = time.time()
        steps = 100
        for _ in range(steps):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / steps * 1000
        print(f"平均推理时间: {avg_time:.2f} ms / step")

        # 打印边界参数信息
        print(f"\n=== Boundary Parameters ===")
        print(f"Lower bound learnable: {model.learnable_lower_bound}")
        print(f"Upper bound learnable: {model.learnable_upper_bound}")
        print(f"Initial lower bound: {model.lower_bound.item():.6f}")
        print(f"Initial upper bound: {model.upper_bound.item():.6f}")

        for size in config['sizes']:
            print(f"\nInput size: {size}")

            # 创建一个随机的溶质场（值在lower_bound和upper_bound之间）
            random_field = torch.rand(size).to(device)
            # 将随机场缩放到界限范围内
            lower_bound_val = model.lower_bound.item()
            upper_bound_val = model.upper_bound.item()
            solute_field = lower_bound_val + (upper_bound_val - lower_bound_val) * random_field

            # 计算初始总溶质量
            initial_total_mass = solute_field.sum().item()
            print(f"Initial total solute mass: {initial_total_mass:.6f}")

            # 检查初始场的最大最小值
            print(f"Initial min: {solute_field.min().item():.6f}, max: {solute_field.max().item():.6f}")
            print(f"Bounds: min={lower_bound_val:.6f}, max={upper_bound_val:.6f}")

            # 记录每个模块的执行时间
            timing_results = {}

            for i in range(50):
                next_field_full, of_change, if_change = model(solute_field)

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

            # 分离并处理outflow和inflow的通量
            outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])
            outflow_distribution = F.softmax(raw_fluxes[:, 1:model.num_neighbors + 1], dim=1)

            inflow_percentage = torch.sigmoid(raw_fluxes[:, model.num_neighbors + 1:model.num_neighbors + 2])
            inflow_distribution = F.softmax(raw_fluxes[:, model.num_neighbors + 2:], dim=1)

            torch.cuda.synchronize()
            timing_results['flux_prediction'] = time.time() - start_time

            # 4. 测量运输计算时间（优化版本）
            start_time = time.time()
            outflow_change, inflow_change = model._compute_transport(
                solute_field,
                outflow_percentage,
                outflow_distribution,
                inflow_percentage,
                inflow_distribution,
                model.lower_bound,
                model.upper_bound
            )
            combined_change = (outflow_change + inflow_change) / 2
            next_field_computed = solute_field + combined_change

            torch.cuda.synchronize()
            timing_results['transport_computation_optimized'] = time.time() - start_time

            # 5. 整体前向传播时间
            start_time = time.time()
            next_field_full, of_change, if_change = model(solute_field)
            torch.cuda.synchronize()
            timing_results['full_forward'] = time.time() - start_time

            # 验证下一时刻场的最大最小值是否在边界范围内
            print(f"Next field min: {next_field_full.min().item():.6f}, max: {next_field_full.max().item():.6f}")

            # 验证守恒性
            final_total_mass = next_field_full.sum().item()
            mass_difference = final_total_mass - initial_total_mass
            print(f"Final total solute mass: {final_total_mass:.6f}")
            print(f"Mass difference: {mass_difference:.8f} ({(mass_difference / initial_total_mass) * 100:.8f}%)")

            # 检查两种方法预测的变化量差异
            approach_diff = torch.abs(of_change - if_change).mean().item()
            print(f"Average difference between outflow and inflow approaches: {approach_diff:.8f}")

            # 输出各模块执行时间
            print("\nPerformance Breakdown:")
            print(f"{'Module':<35} {'Time (ms)':<12} {'Percentage':<10}")
            print("-" * 60)
            full_time = timing_results['full_forward'] * 1000  # 转换为毫秒
            for module, t in timing_results.items():
                ms_time = t * 1000  # 转换为毫秒
                percentage = (ms_time / full_time) * 100
                print(f"{module:<35} {ms_time:<12.2f} {percentage:<10.2f}%")

            # 测试多步迭代的守恒性和边界保证
            print("\n=== Testing multi-step conservation and bounds ===")
            current_field = solute_field.clone()
            num_steps = 10
            step_masses = []
            min_values = []
            max_values = []

            # Test in eval mode (bounds should be fixed)
            model.eval()
            print(
                f"Eval mode - Lower bound: {model.lower_bound.item():.6f}, Upper bound: {model.upper_bound.item():.6f}")

            for step in range(num_steps):
                with torch.no_grad():
                    current_field, _, _ = model(current_field)
                    step_mass = current_field.sum().item()
                    step_min = current_field.min().item()
                    step_max = current_field.max().item()

                    step_masses.append(step_mass)
                    min_values.append(step_min)
                    max_values.append(step_max)

                    mass_diff_pct = (step_mass - initial_total_mass) / initial_total_mass * 100
                    print(f"Step {step + 1}: Mass = {step_mass:.6f}, Diff = {mass_diff_pct:.8f}%, "
                          f"Min = {step_min:.6f}, Max = {step_max:.6f}")

            # 计算多步迭代的最大质量变化率
            max_diff_pct = max([abs((m - initial_total_mass) / initial_total_mass * 100) for m in step_masses])
            print(f"Maximum mass difference over {num_steps} steps: {max_diff_pct:.8f}%")

            # 检查是否所有步骤都遵守了边界条件
            bounds_violated = any(v < lower_bound_val for v in min_values) or any(
                v > upper_bound_val for v in max_values)
            if bounds_violated:
                print("WARNING: Bounds were violated during multi-step simulation!")
            else:
                print(
                    f"All values remained within bounds [{lower_bound_val:.6f}, {upper_bound_val:.6f}] during simulation.")

            # 返回训练模式测试边界参数的可学习性
            model.train()
            print("\n=== Testing learnable bounds in train mode ===")

            # 模拟一个简单的优化步骤
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 保存初始边界值
            initial_lower = model.lower_bound.item()
            initial_upper = model.upper_bound.item()

            # 做一个前向传播和反向传播
            output, _, _ = model(solute_field)
            loss = output.mean()  # 简单的损失函数
            loss.backward()
            optimizer.step()

            # 检查边界是否更新
            updated_lower = model.lower_bound.item()
            updated_upper = model.upper_bound.item()

            print(f"Before optimization - Lower: {initial_lower:.6f}, Upper: {initial_upper:.6f}")
            print(f"After optimization  - Lower: {updated_lower:.6f}, Upper: {updated_upper:.6f}")
            print(f"Lower bound change: {updated_lower - initial_lower:.8f}")
            print(f"Upper bound change: {updated_upper - initial_upper:.8f}")

            # 可视化通量特性
            model.eval()
            with torch.no_grad():
                raw_fluxes = model.flux_conv(features)

                # Outflow approach
                outflow_percentage = torch.sigmoid(raw_fluxes[:, 0:1])
                outflow_distribution = F.softmax(raw_fluxes[:, 1:model.num_neighbors + 1], dim=1)

                # Inflow approach
                inflow_percentage = torch.sigmoid(raw_fluxes[:, model.num_neighbors + 1:model.num_neighbors + 2])
                inflow_distribution = F.softmax(raw_fluxes[:, model.num_neighbors + 2:], dim=1)

                # 统计信息
                print("\n=== Flux Statistics ===")

                # 1. Outflow 百分比统计
                outflow_stats = {
                    'mean': outflow_percentage.mean().item(),
                    'min': outflow_percentage.min().item(),
                    'max': outflow_percentage.max().item(),
                    'std': outflow_percentage.std().item(),
                }
                print("\nOutflow Percentage Statistics:")
                for stat, value in outflow_stats.items():
                    print(f"{stat}: {value:.6f}")

                # 2. Inflow 百分比统计
                inflow_stats = {
                    'mean': inflow_percentage.mean().item(),
                    'min': inflow_percentage.min().item(),
                    'max': inflow_percentage.max().item(),
                    'std': inflow_percentage.std().item(),
                }
                print("\nInflow Percentage Statistics:")
                for stat, value in inflow_stats.items():
                    print(f"{stat}: {value:.6f}")

                # 3. Outflow分布比例统计
                outflow_dist_stats = {
                    'mean': outflow_distribution.mean().item(),
                    'min': outflow_distribution.min().item(),
                    'max': outflow_distribution.max().item(),
                    'std': outflow_distribution.std().item(),
                }
                print("\nOutflow Distribution Ratio Statistics:")
                for stat, value in outflow_dist_stats.items():
                    print(f"{stat}: {value:.6f}")

                # 4. Inflow分布比例统计
                inflow_dist_stats = {
                    'mean': inflow_distribution.mean().item(),
                    'min': inflow_distribution.min().item(),
                    'max': inflow_distribution.max().item(),
                    'std': inflow_distribution.std().item(),
                }
                print("\nInflow Distribution Ratio Statistics:")
                for stat, value in inflow_dist_stats.items():
                    print(f"{stat}: {value:.6f}")

                # 5. 计算分布的稀疏性
                outflow_sparsity = 100.0 * (
                        1.0 - (outflow_distribution > 0.01).sum().item() / outflow_distribution.numel())
                inflow_sparsity = 100.0 * (
                        1.0 - (inflow_distribution > 0.01).sum().item() / inflow_distribution.numel())
                print(f"\nOutflow distribution sparsity: {outflow_sparsity:.2f}% (values < 1%)")
                print(f"Inflow distribution sparsity: {inflow_sparsity:.2f}% (values < 1%)")

                # 6. 两种方法的差异
                change_diff_mean = torch.abs(of_change - if_change).mean().item()
                change_diff_max = torch.abs(of_change - if_change).max().item()
                print(f"\nDifference between approaches: mean={change_diff_mean:.6f}, max={change_diff_max:.6f}")

                # 7. 检查流量守恒性的详细统计
                print("\nDetailed Conservation Check:")
                print(f"Initial mass: {initial_total_mass:.6f}")
                print(f"Final mass: {final_total_mass:.6f}")
                print(f"Conservation error: {abs(final_total_mass - initial_total_mass):.10f}")

                # 8. 验证是否需要clamp操作
                percent_out_of_bounds = (torch.logical_or(
                    next_field_full < model.lower_bound,
                    next_field_full > model.upper_bound
                )).float().mean().item() * 100
                print(f"\nPercentage of values out of bounds: {percent_out_of_bounds:.2f}%")