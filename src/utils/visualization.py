import matplotlib
matplotlib.use('Agg') # 强制使用无头模式，只生成文件不弹窗
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def create_wr_custom_colormap():
    """
    创建一个自定义的红-蓝-白颜色映射，数值0为白色，-1为蓝色，1为红色。

    :return: 自定义的颜色映射
    """
    # 颜色映射：白色 -> 红色
    colors = ["white", "red"]
    cmap = LinearSegmentedColormap.from_list("white_red", colors, N=256)
    return cmap

def create_bwr_custom_colormap():
    """
    创建一个自定义的红-蓝-白颜色映射，数值0为白色，-1为蓝色，1为红色。

    :return: 自定义的颜色映射
    """
    # 颜色映射：蓝色 -> 白色 -> 红色
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    return cmap


def model_performance_during_training(current_phi, outflow_change, inflow_change, predicted_phi, next_phi, epoch, loss,
                                      mode, mpv_dir):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax1, ax2, ax3, ax4 = ax.flatten()

    # 获取数据
    of_change = outflow_change.cpu().detach().numpy()
    if_change = inflow_change.cpu().detach().numpy()
    dp_scaled = predicted_phi.cpu().detach().numpy() - current_phi.squeeze().cpu().detach().numpy()
    t0 = next_phi.squeeze().cpu().detach().numpy() - current_phi.squeeze().cpu().detach().numpy()

    # 计算共同的颜色范围
    max_value = max(np.max(np.abs(of_change)), np.max(np.abs(if_change)), np.max(np.abs(t0)))

    # 绘制四张图
    im1 = ax1.imshow(of_change[0], cmap=create_bwr_custom_colormap(), vmin=-max_value, vmax=max_value)
    im2 = ax2.imshow(if_change[0], cmap=create_bwr_custom_colormap(), vmin=-max_value, vmax=max_value)
    im3 = ax3.imshow(dp_scaled[0], cmap=create_bwr_custom_colormap(), vmin=-max_value, vmax=max_value)
    im4 = ax4.imshow(t0[0], cmap=create_bwr_custom_colormap(), vmin=-max_value, vmax=max_value)

    plt.colorbar(im1, ax=ax1, shrink=0.7)
    plt.colorbar(im2, ax=ax2, shrink=0.7)
    plt.colorbar(im3, ax=ax3, shrink=0.7)
    plt.colorbar(im4, ax=ax4, shrink=0.7)

    ax1.set_title(f"Outflow Change [0]")
    ax2.set_title(f"Inflow Change [0]")
    ax3.set_title(f"Predicted dPhi [0]")
    ax4.set_title(f"True dPhi [0]")

    plt.suptitle(f"{mode}: epoch:{epoch + 1}    {mode} loss:{loss:.3e}", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(mpv_dir, f"{mode} prediction_epoch{epoch}.png"))
    plt.close()

def plot_train_loss_curve(loss_data, file_path, file_name):
    """
    绘制训练损失曲线

    支持新旧两种数据格式:
    - 旧格式: tuple of (train, test, dcl, pred, best, lr)
    - 新格式: dict with keys
    """
    # 兼容旧格式
    if isinstance(loss_data, (list, tuple)) and len(loss_data) == 6:
        train_losses, test_losses, test_dcl_losses, test_p_losses, best_losses, optimizer_lrs = loss_data
    else:
        # 新格式
        train_losses = loss_data.get('train_losses', [])
        test_losses = loss_data.get('val_losses', [])
        test_dcl_losses = loss_data.get('val_losses_dict', {}).get('dcl_loss', [])
        test_p_losses = loss_data.get('val_losses_dict', {}).get('p_loss', [])
        best_losses = loss_data.get('best_losses', [])
        optimizer_lrs = loss_data.get('optimizer_lrs', [])

    # 创建两个子图：一个用于总损失，一个用于细分损失
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 第一个子图：总损失和学习率
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(test_losses, label="Test Loss")
    ax1.plot(best_losses, label="Best Loss")

    # 在右侧Y轴上显示学习率
    ax1_lr = ax1.twinx()
    ax1_lr.plot(optimizer_lrs, label="Learning Rate", color='purple', linestyle='--')
    ax1_lr.set_ylabel("Learning Rate", color='purple')
    ax1_lr.tick_params(axis='y', labelcolor='purple')

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f'Total Loss: {file_name}   Best Loss: {np.min(best_losses) if best_losses else 0:.3e}')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1_lr.legend(loc='upper right')

    # 第二个子图：细分损失
    ax2.plot(test_losses, label="Total Test Loss", color='blue')
    if test_dcl_losses:
        ax2.plot(test_dcl_losses, label="DCL Loss (Dual Consistency)", color='green')
    if test_p_losses:
        ax2.plot(test_p_losses, label="Prediction Loss", color='red')

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title('Component Losses')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(file_path + '/loss_curve.png')
    plt.close()

def model_performance_visualization(data_list, plot_config, evaluation_mode, t, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (data, title, config_key) in zip(axes.flat, data_list):
        im = ax.imshow(data, cmap=plot_config[config_key]["cmap"],
                       vmin=plot_config[config_key]["vmin"],
                       vmax=plot_config[config_key]["vmax"])
        ax.set_title(title, fontsize=20)
        plt.colorbar(im, ax=ax, shrink=0.7)
    # plt.suptitle(f"{output_dir.split('/')[1].title()}  {evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    plt.suptitle(f"{evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    plt.tight_layout()
    # print(os.path.join(output_dir, f"step_{t:012d}.png"))
    plt.savefig(os.path.join(output_dir, f"step_{t:012d}.png"))
    # plt.show()
    plt.close()

def model_performance_visualization13(data_list, plot_config, evaluation_mode, t, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (data, title, config_key) in zip(axes.flat, data_list):
        # im = ax.imshow(data, cmap=plot_config[config_key]["cmap"],
        #                vmin=0,
        #                vmax=0.3)
        im = ax.imshow(data, cmap=plot_config[config_key]["cmap"],
                       vmin=plot_config[config_key]["vmin"],
                       vmax=plot_config[config_key]["vmax"])
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        # 设置颜色条刻度标签的字体大小
        cbar.ax.tick_params(labelsize=20)
        # ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # plt.colorbar(im, ax=ax, shrink=0.7)
    # plt.suptitle(f"{output_dir.split('/')[1].title()}  {evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    # plt.suptitle(f"{evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    plt.tight_layout()
    # print(os.path.join(output_dir, f"step_{t:012d}.png"))
    plt.savefig(os.path.join(output_dir, f"step13_{t:012d}.png"))
    # plt.show()
    plt.close()

def model_performance_visualization11(data_list, plot_config, evaluation_mode, t, output_dir):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    for (data, title, config_key) in data_list:
        im = axes.imshow(data, cmap=plot_config[config_key]["cmap"],
                       vmin=plot_config[config_key]["vmin"],
                       vmax=plot_config[config_key]["vmax"])
        # axes.set_title(title, fontsize=20)
        # plt.colorbar(im, ax=ax, shrink=0.7)
    # plt.suptitle(f"{output_dir.split('/')[1].title()}  {evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    # plt.suptitle(f"{evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    axes.axis('off')
    plt.tight_layout()
    # print(os.path.join(output_dir, f"step_{t:012d}.png"))
    plt.savefig(os.path.join(output_dir, f"step11_{evaluation_mode}_{t:012d}.png"))
    # plt.show()
    plt.close()

def delta_phi_visualization(data_list, plot_config, evaluation_mode, t, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (data, title, config_key) in zip(axes.flat, data_list):
        im = ax.imshow(data, cmap=plot_config[config_key]["cmap"],
                       vmin=plot_config[config_key]["vmin"],
                       vmax=plot_config[config_key]["vmax"])
        # ax.set_title(title, fontsize=20)
        plt.colorbar(im, ax=ax, shrink=0.7)
    # plt.suptitle(f"{output_dir.split('/')[1].title()}  {evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    # plt.suptitle(f"{evaluation_mode.title()}  step_{t:012d}", fontsize=25)
    plt.tight_layout()
    # print(os.path.join(output_dir, f"step_{t:012d}.png"))
    plt.savefig(os.path.join(output_dir, f"dp_step_{t:012d}.png"))
    # plt.show()
    plt.close()


def visualize_conservation(conservation_true, conservation_pred, output_dir, evaluation_mode):
    """
    可视化守恒性随时间的变化

    参数:
    conservation_true: 真实的总和值
    conservation_pred: 预测的总和值
    output_dir: 输出目录
    evaluation_mode: 评估模式
    """
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6.5))

    plt.plot(conservation_pred, '-', color='red', linewidth=4, label='Predicted')

    plt.plot(conservation_true, '--', color='blue', linewidth=4, label='True')


    # 计算相对误差
    avg_deviation = np.mean(np.abs(np.array(conservation_pred) - np.array(conservation_true))) / np.mean(
        np.abs(conservation_true)) * 100

    # 设置标题和标签
    # plt.title(f'Conservation Analysis ({evaluation_mode}) - Avg Deviation: {avg_deviation:.2f}%', fontsize=14)
    plt.xlabel('Time Step', fontsize=25)
    plt.ylabel('Total Sum of Field Values', fontsize=25)
    plt.ylim(np.mean(np.array(conservation_true))*0.9,np.mean(np.array(conservation_true))*1.1)

    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18) # 通常x轴刻度也需要调整


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=22)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'conservation_{evaluation_mode}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 1D 可视化函数 ====================

def plot_1d_prediction_comparison(
    x_coords: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    time_step: int,
    output_path: str,
    title: str = None
):
    """
    1D 预测 vs 真实曲线对比

    Args:
        x_coords: 空间坐标
        true_values: 真实值
        pred_values: 预测值
        time_step: 时间步
        output_path: 保存路径
        title: 标题
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: 预测 vs 真实
    ax1.plot(x_coords, true_values, 'b-', linewidth=2, label='True')
    ax1.plot(x_coords, pred_values, 'r--', linewidth=2, label='Predicted')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(f'{title} - Step {time_step}', fontsize=14)

    # 下图: 绝对误差
    abs_error = np.abs(pred_values - true_values)
    ax2.plot(x_coords, abs_error, 'g-', linewidth=2)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'MAE: {np.mean(abs_error):.6e}', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_1d_spacetime_heatmap(
    true_data: np.ndarray,
    pred_data: np.ndarray,
    output_path: str,
    title: str = None
):
    """
    1D 时空热力图对比

    Args:
        true_data: 真实数据 [time, space]
        pred_data: 预测数据 [time, space]
        output_path: 保存路径
        title: 标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 共同的颜色范围
    vmin = min(true_data.min(), pred_data.min())
    vmax = max(true_data.max(), pred_data.max())

    # 真实值
    im1 = axes[0].imshow(true_data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('True', fontsize=14)
    axes[0].set_xlabel('Space', fontsize=12)
    axes[0].set_ylabel('Time', fontsize=12)
    plt.colorbar(im1, ax=axes[0])

    # 预测值
    im2 = axes[1].imshow(pred_data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted', fontsize=14)
    axes[1].set_xlabel('Space', fontsize=12)
    plt.colorbar(im2, ax=axes[1])

    # 误差
    error = np.abs(pred_data - true_data)
    im3 = axes[2].imshow(error, aspect='auto', cmap='hot')
    axes[2].set_title(f'Absolute Error (Mean: {np.mean(error):.4e})', fontsize=14)
    axes[2].set_xlabel('Space', fontsize=12)
    plt.colorbar(im3, ax=axes[2])

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_1d_error_curve(
    errors: list,
    output_path: str,
    title: str = None,
    ylabel: str = 'Error'
):
    """
    1D 误差随时间变化曲线

    Args:
        errors: 误差列表
        output_path: 保存路径
        title: 标题
        ylabel: Y轴标签
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errors, linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 浅水方程可视化 ====================

def plot_shallow_water_fields(
    true_h: np.ndarray,
    pred_h: np.ndarray,
    true_mx: np.ndarray,
    pred_mx: np.ndarray,
    true_my: np.ndarray,
    pred_my: np.ndarray,
    output_path: str,
    time_step: int
):
    """
    浅水方程三通道联合可视化 (3x3 子图布局)

    Args:
        true_h, pred_h: 水深 (真实/预测)
        true_mx, pred_mx: x方向动量
        true_my, pred_my: y方向动量
        output_path: 保存路径
        time_step: 时间步
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    fields = [
        ('h', true_h, pred_h),
        ('mx', true_mx, pred_mx),
        ('my', true_my, pred_my)
    ]

    for row, (name, true_field, pred_field) in enumerate(fields):
        # 共同颜色范围
        vmin = min(true_field.min(), pred_field.min())
        vmax = max(true_field.max(), pred_field.max())

        # 真实值
        im1 = axes[row, 0].imshow(true_field, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f'True {name}', fontsize=12)
        plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)

        # 预测值
        im2 = axes[row, 1].imshow(pred_field, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'Predicted {name}', fontsize=12)
        plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)

        # 误差
        error = np.abs(pred_field - true_field)
        im3 = axes[row, 2].imshow(error, cmap='hot')
        axes[row, 2].set_title(f'Error {name} (MAE: {np.mean(error):.4e})', fontsize=12)
        plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)

    fig.suptitle(f'Shallow Water Fields - Step {time_step}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_field_to_tecplot(
    field: np.ndarray,
    output_path: str,
    field_name: str = 'phi'
):
    """
    保存2D场为Tecplot .dat格式

    Args:
        field: 2D数组
        output_path: 输出路径
        field_name: 场名称
    """
    if len(field.shape) != 2:
        raise ValueError(f"Field must be 2D, got shape {field.shape}")

    I, J = field.shape

    with open(output_path, 'w') as f:
        f.write(f'VARIABLE="x","y","{field_name}"\n')
        f.write(f'ZONE t="BIG ZONE", I={I}, J={J}, F=POINT\n')

        for value in field.flatten():
            f.write(f"{value:.6f}\n")


# ==================== 统计曲线可视化 ====================

def plot_violation_curve(
    violation_rates: list,
    output_path: str,
    bound_type: str = 'lower',
    title: str = None
):
    """
    越界率随时间变化曲线

    Args:
        violation_rates: 越界率列表 (百分比)
        output_path: 保存路径
        bound_type: 'lower', 'upper', 或 'both'
        title: 标题
    """
    plt.figure(figsize=(10, 6))

    if bound_type == 'both' and isinstance(violation_rates, dict):
        plt.plot(violation_rates['lower'], label='Lower Bound Violation', linewidth=2)
        plt.plot(violation_rates['upper'], label='Upper Bound Violation', linewidth=2)
        plt.legend()
    else:
        plt.plot(violation_rates, linewidth=2)

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Violation Rate (%)', fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'{bound_type.capitalize()} Bound Violation Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rollout_error_vs_horizon(
    errors_by_horizon: dict,
    output_path: str,
    title: str = None
):
    """
    Rollout 误差 vs 预测步数

    Args:
        errors_by_horizon: {horizon: error} 字典
        output_path: 保存路径
        title: 标题
    """
    horizons = sorted(errors_by_horizon.keys())
    errors = [errors_by_horizon[h] for h in horizons]

    plt.figure(figsize=(10, 6))
    plt.plot(horizons, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Prediction Horizon (steps)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Rollout Error vs Prediction Horizon', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pushforward_loss_breakdown(
    step_losses: dict,
    output_path: str,
    title: str = None
):
    """
    Pushforward 训练中各步损失分解曲线

    Args:
        step_losses: {step_k: [loss_per_epoch]} 字典
        output_path: 保存路径
        title: 标题
    """
    plt.figure(figsize=(12, 6))

    for step, losses in sorted(step_losses.items()):
        plt.plot(losses, label=f'Step {step}', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Pushforward Loss Breakdown', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

