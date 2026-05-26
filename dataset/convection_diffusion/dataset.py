"""
一维对流-扩散方程数据集生成器
============================================================
方程: ∂c/∂t + u·∂c/∂x = D·∂²c/∂x²

符号说明:
    c - 浓度场 (concentration)，场变量
    u - 对流速度 (velocity)，常数
    D - 扩散系数 (diffusion coefficient)

采用傅里叶谱方法求解，周期性边界条件
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# ============================================================
# 全局参数设置
# ============================================================
L = 1.0  # 空间域长度
N = 32  # 空间网格点数
D = 0.005  # 扩散系数
T_final = 5.0  # 总时间
dt = 0.1  # 时间步长

# 计算时间步数
Nt = int(T_final / dt)  # = 50

# 空间离散
x = np.linspace(0, L, N, endpoint=False)  # 周期边界，不含右端点
dx = L / N

# 波数（傅里叶空间）
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

# 数据集参数
N_train = 100  # 训练集样本数
N_val = 10  # 验证集样本数
N_test = 10  # 测试集样本数

# 速度范围
u_min = 0.0
u_max = 0.2

# 全局随机种子（保证可复现性）
GLOBAL_SEED = 42

# 输出目录
BASE_DIR = "./convection_diffusion_dataset"


# ============================================================
# 求解器函数
# ============================================================
def generate_initial_condition(rng, n_waves=4):
    """
    生成随机初始条件：多个低频正弦波叠加
    确保结果始终在[0, 1]范围内（通过控制振幅总和实现，非截断方式）

    策略：
        基础偏移为0.5，控制所有正弦波振幅之和不超过0.5
        这样 c = 0.5 + sum(a_i * sin(...)) 中：
        - 最小值 >= 0.5 - 0.5 = 0
        - 最大值 <= 0.5 + 0.5 = 1

    Parameters
    ----------
    rng : numpy.random.Generator
        随机数生成器
    n_waves : int
        叠加的正弦波数量

    Returns
    -------
    c0 : ndarray
        初始浓度场，形状 (N,)，值域 [0, 1]
    """
    # 基础偏移（居中）
    base = 0.5

    # 目标振幅总和（确保 base - total_amp >= 0 且 base + total_amp <= 1）
    # 即 total_amp <= 0.5，这里留一点余量
    max_total_amplitude = 0.45
    target_total_amplitude = rng.uniform(0.2, max_total_amplitude)

    # 生成随机振幅权重并归一化，使振幅总和等于目标值
    raw_weights = rng.uniform(0.5, 1.0, size=n_waves)
    amplitudes = raw_weights / raw_weights.sum() * target_total_amplitude

    # 初始化浓度场
    c = np.ones(N) * base

    for i in range(n_waves):
        phase = rng.uniform(0, 2 * np.pi)  # 随机相位
        freq = rng.integers(1, 5)  # 随机频率 (1-4个完整周期)
        c += amplitudes[i] * np.sin(2 * np.pi * freq * x / L + phase)

    return c


def solve_convection_diffusion(c0, u_velocity, D_coeff, dt_step, n_steps):
    """
    使用傅里叶谱方法求解一维对流-扩散方程

    方程: ∂c/∂t + u·∂c/∂x = D·∂²c/∂x²

    傅里叶变换后:
        dĉ_k/dt = -(i·u·k + D·k²)·ĉ_k

    精确解:
        ĉ_k(t+Δt) = ĉ_k(t)·exp[-(i·u·k + D·k²)·Δt]

    Parameters
    ----------
    c0 : ndarray
        初始浓度场，形状 (N,)
    u_velocity : float
        对流速度（常数）
    D_coeff : float
        扩散系数
    dt_step : float
        时间步长
    n_steps : int
        时间步数

    Returns
    -------
    c_history : ndarray
        浓度场演化序列，形状 (n_steps+1, N)
    """
    # 存储所有时间步的结果
    c_history = np.zeros((n_steps + 1, N))
    c_history[0] = c0.copy()

    # 初始傅里叶变换
    c_hat = np.fft.fft(c0)

    # 积分因子 (精确积分，无条件稳定)
    # exp[-(i·u·k + D·k²)·Δt]
    integrating_factor = np.exp(-(1j * u_velocity * k + D_coeff * k ** 2) * dt_step)

    # 时间推进
    for n in range(1, n_steps + 1):
        # 傅里叶空间精确时间推进
        c_hat = c_hat * integrating_factor
        # 逆变换回物理空间
        c = np.real(np.fft.ifft(c_hat))
        c_history[n] = c

    return c_history


def save_to_h5(filepath, c_data, u_velocity):
    """
    保存单个样本数据到HDF5文件

    Parameters
    ----------
    filepath : str
        保存路径
    c_data : ndarray
        浓度场演化序列，形状 (Nt+1, N)
    u_velocity : float
        对流速度值
    """
    with h5py.File(filepath, 'w') as f:
        # 保存浓度场序列 c(t, x)
        # 形状: (Nt+1, N) = (51, 32)
        f.create_dataset('c', data=c_data.astype(np.float32))

        # 保存速度场（与单时刻浓度场形状相同的常数数组）
        # 形状: (N,) = (32,)
        # 用于深度学习时与某时刻的c拼接
        u_field = np.full(N, u_velocity, dtype=np.float32)
        f.create_dataset('u', data=u_field)

        # 保存元数据（方便后续检查）
        meta = f.create_group('metadata')
        meta.attrs['L'] = L
        meta.attrs['N'] = N
        meta.attrs['D'] = D
        meta.attrs['dt'] = dt
        meta.attrs['T_final'] = T_final
        meta.attrs['Nt'] = Nt
        meta.attrs['u_velocity'] = u_velocity


# ============================================================
# 数据集生成
# ============================================================
def generate_dataset(output_dir, n_samples, rng, dataset_name):
    """
    生成一个数据集（训练/验证/测试）

    Parameters
    ----------
    output_dir : str
        输出文件夹路径
    n_samples : int
        样本数量
    rng : numpy.random.Generator
        随机数生成器
    dataset_name : str
        数据集名称（用于打印信息）
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"生成 {dataset_name} 数据集 ({n_samples} 个样本)")
    print(f"保存目录: {output_dir}")
    print(f"{'=' * 50}")

    for i in range(n_samples):
        # 随机生成对流速度 u ∈ [0, 0.2]
        u_velocity = rng.uniform(u_min, u_max)

        # 随机生成初始条件
        c0 = generate_initial_condition(rng)

        # 求解对流-扩散方程
        c_history = solve_convection_diffusion(c0, u_velocity, D, dt, Nt)

        # 保存到h5文件
        filepath = os.path.join(output_dir, f"sample_{i:04d}.h5")
        save_to_h5(filepath, c_history, u_velocity)

        # 打印进度
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == n_samples:
            print(f"  进度: {i + 1:3d}/{n_samples}, u = {u_velocity:.4f}")

    print(f"{dataset_name} 数据集生成完成!\n")


# ============================================================
# 可视化检查函数
# ============================================================
def visualize_h5_sample(h5_filepath, save_dir, sample_name):
    """
    读取h5文件并生成三张可视化图片

    图1: 时空演化云图
    图2: 不同时刻的浓度曲线
    图3: 质量守恒检验

    Parameters
    ----------
    h5_filepath : str
        h5文件路径
    save_dir : str
        图片保存目录
    sample_name : str
        样本名称（用于文件命名）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    with h5py.File(h5_filepath, 'r') as f:
        c_data = f['c'][:]  # (Nt+1, N)
        u_field = f['u'][:]  # (N,)
        u_velocity = f['metadata'].attrs['u_velocity']
        N_grid = f['metadata'].attrs['N']
        dt_val = f['metadata'].attrs['dt']
        L_val = f['metadata'].attrs['L']
        D_val = f['metadata'].attrs['D']

    # 重建网格
    n_times = c_data.shape[0]
    x_grid = np.linspace(0, L_val, N_grid, endpoint=False)
    t_grid = np.arange(n_times) * dt_val
    dx_val = L_val / N_grid

    # 计算质量 (总浓度)
    mass = np.sum(c_data, axis=1) * dx_val

    # ==================== 图1: 时空演化云图 ====================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)

    # 设置合适的等高线级数
    c_min, c_max = c_data.min(), c_data.max()
    levels = np.linspace(c_min, c_max, 50)

    cf = ax1.contourf(T_mesh, X_mesh, c_data.T, levels=levels, cmap='RdBu_r')
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label('Concentration c(x,t)', fontsize=12)

    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Position x', fontsize=12)
    ax1.set_title(f'Spatiotemporal Evolution\n(u = {u_velocity:.4f}, D = {D_val})',
                  fontsize=14)

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, f'{sample_name}_01_contour.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ==================== 图2: 不同时刻的曲线 ====================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # 选择6个时刻展示
    n_curves = 6
    time_indices = np.linspace(0, n_times - 1, n_curves, dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 0.9, n_curves))

    for idx, color in zip(time_indices, colors):
        ax2.plot(x_grid, c_data[idx], color=color, linewidth=2,
                 label=f't = {t_grid[idx]:.1f}')

    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Concentration c(x,t)', fontsize=12)
    ax2.set_title(f'Concentration Profiles at Different Times\n(u = {u_velocity:.4f}, D = {D_val})',
                  fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, L_val])

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, f'{sample_name}_02_profiles.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ==================== 图3: 质量守恒检验 ====================
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # 绘制质量随时间变化
    ax3.plot(t_grid, mass, 'b-', linewidth=2, label='Computed mass')
    ax3.axhline(y=mass[0], color='r', linestyle='--', linewidth=1.5,
                label=f'Initial mass = {mass[0]:.6f}')

    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Total Mass: ∫c dx', fontsize=12)
    ax3.set_title(f'Mass Conservation Check\n(u = {u_velocity:.4f}, D = {D_val})',
                  fontsize=14)

    # 放大y轴范围查看变化
    ax3.set_ylim([mass[0] - 0.001, mass[0] + 0.001])

    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 添加相对误差（次坐标轴）
    ax3_twin = ax3.twinx()
    rel_error = np.abs(mass - mass[0]) / (np.abs(mass[0]) + 1e-16)
    rel_error_safe = np.maximum(rel_error, 1e-16)  # 避免log(0)
    ax3_twin.semilogy(t_grid, rel_error_safe, 'g--', linewidth=1.5,
                      alpha=0.7, label='Relative error')
    ax3_twin.set_ylabel('Relative Error', fontsize=12, color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.legend(loc='upper right', fontsize=10)

    # 添加误差统计信息
    max_error = np.max(rel_error)
    textstr = f'Max relative error: {max_error:.2e}'
    ax3.text(0.02, 0.02, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, f'{sample_name}_03_mass_conservation.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)


def visualize_all_datasets(base_dir, fig_dir, n_samples_per_dataset=3):
    """
    对所有数据集进行可视化检查

    Parameters
    ----------
    base_dir : str
        数据集根目录
    fig_dir : str
        图片保存目录
    n_samples_per_dataset : int
        每个数据集检查的样本数
    """
    datasets = ['train', 'val', 'test']

    print("\n" + "=" * 60)
    print("开始可视化检查")
    print(f"图片保存目录: {fig_dir}")
    print("=" * 60)

    for dataset_name in datasets:
        dataset_dir = os.path.join(base_dir, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"警告: 目录不存在 - {dataset_dir}")
            continue

        # 获取所有h5文件
        h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])

        if len(h5_files) == 0:
            print(f"警告: 目录为空 - {dataset_dir}")
            continue

        # 均匀选择要检查的样本
        n_to_check = min(n_samples_per_dataset, len(h5_files))
        check_indices = np.linspace(0, len(h5_files) - 1, n_to_check, dtype=int)

        print(f"\n检查 {dataset_name} 数据集 (共 {len(h5_files)} 个文件, 检查 {n_to_check} 个)")

        for idx in check_indices:
            h5_file = h5_files[idx]
            h5_path = os.path.join(dataset_dir, h5_file)
            sample_name = f"{dataset_name}_{h5_file.replace('.h5', '')}"

            print(f"  正在处理: {dataset_name}/{h5_file}")
            visualize_h5_sample(h5_path, fig_dir, sample_name)

    print(f"\n可视化检查完成! 共生成 {len(datasets) * n_samples_per_dataset * 3} 张图片")
    print(f"图片保存在: {fig_dir}")


def inspect_h5_structure(h5_filepath):
    """
    打印h5文件的数据结构，包括数值范围检查

    Parameters
    ----------
    h5_filepath : str
        h5文件路径
    """
    print(f"\n{'=' * 50}")
    print(f"检查 HDF5 文件结构: {h5_filepath}")
    print(f"{'=' * 50}")

    with h5py.File(h5_filepath, 'r') as f:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: '{name}'")
                print(f"    - Shape: {obj.shape}")
                print(f"    - Dtype: {obj.dtype}")
                print(f"    - Size: {obj.size}")
                # 数值范围检查
                data = obj[:]
                print(f"    - Min value: {data.min():.6f}")
                print(f"    - Max value: {data.max():.6f}")
                print(f"    - Mean value: {data.mean():.6f}")
                print(f"    - Std value: {data.std():.6f}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: '{name}'")
                for key, val in obj.attrs.items():
                    print(f"    - Attr '{key}': {val}")

        print("Root datasets and groups:")
        f.visititems(print_item)

    print(f"{'=' * 50}\n")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":

    # 打印参数信息
    print("=" * 60)
    print("一维对流-扩散方程数据集生成器")
    print("=" * 60)
    print("方程: ∂c/∂t + u·∂c/∂x = D·∂²c/∂x²")
    print("-" * 60)
    print("参数设置:")
    print(f"  空间域: [0, {L}]")
    print(f"  空间网格点数 N = {N}")
    print(f"  空间步长 dx = {dx:.6f}")
    print(f"  时间域: [0, {T_final}]")
    print(f"  时间步长 dt = {dt}")
    print(f"  时间步数 Nt = {Nt}")
    print(f"  扩散系数 D = {D}")
    print(f"  速度范围 u ∈ [{u_min}, {u_max}]")
    print("-" * 60)
    print("数据集规模:")
    print(f"  训练集: {N_train} 个样本")
    print(f"  验证集: {N_val} 个样本")
    print(f"  测试集: {N_test} 个样本")
    print("-" * 60)
    print(f"每个样本数据形状:")
    print(f"  c (浓度场序列): ({Nt + 1}, {N}) = ({Nt + 1}, {N})")
    print(f"  u (速度场): ({N},)")
    print(f"全局随机种子: {GLOBAL_SEED}")
    print("=" * 60)

    # 创建目录结构
    train_dir = os.path.join(BASE_DIR, "train")
    val_dir = os.path.join(BASE_DIR, "val")
    test_dir = os.path.join(BASE_DIR, "test")
    fig_dir = os.path.join(BASE_DIR, "figures")

    # 设置主随机数生成器
    master_rng = np.random.default_rng(GLOBAL_SEED)

    # 为每个数据集生成独立的种子（确保可复现性）
    train_seed = master_rng.integers(0, 2 ** 31)
    val_seed = master_rng.integers(0, 2 ** 31)
    test_seed = master_rng.integers(0, 2 ** 31)

    print(f"\n子种子分配:")
    print(f"  Train seed: {train_seed}")
    print(f"  Val seed: {val_seed}")
    print(f"  Test seed: {test_seed}")

    # ==================== 生成数据集 ====================
    generate_dataset(train_dir, N_train, np.random.default_rng(train_seed), "Train")
    generate_dataset(val_dir, N_val, np.random.default_rng(val_seed), "Validation")
    generate_dataset(test_dir, N_test, np.random.default_rng(test_seed), "Test")

    # ==================== 检查h5文件结构 ====================
    sample_h5 = os.path.join(train_dir, "sample_0000.h5")
    if os.path.exists(sample_h5):
        inspect_h5_structure(sample_h5)

    # ==================== 可视化检查 ====================
    visualize_all_datasets(BASE_DIR, fig_dir, n_samples_per_dataset=100)

    # ==================== 最终统计 ====================
    print("\n" + "=" * 60)
    print("数据集生成完成!")
    print("=" * 60)
    print(f"数据保存位置: {os.path.abspath(BASE_DIR)}")
    print(f"目录结构:")
    print(f"  {BASE_DIR}/")
    print(f"  ├── train/     ({N_train} 个 .h5 文件)")
    print(f"  ├── val/       ({N_val} 个 .h5 文件)")
    print(f"  ├── test/      ({N_test} 个 .h5 文件)")
    print(f"  └── figures/   (可视化检查图片)")
    print("=" * 60)
