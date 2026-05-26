"""
Optimized Effective Receptive Field Analysis for FluxNet_D Models

This script analyzes the effective receptive field (ERF) of FluxNet_D models
trained with different temporal resolutions (10dt, 100dt, 1000dt).

Optimizations:
1. Multi-GPU parallel processing for different models
2. Single forward pass per sample point, compute all channel gradients
3. Parallel processing of sample points using multiprocessing
4. Vectorized numpy operations

Output:
- NPY files for each channel's RF map
- Statistics text file
- Markdown table with RF size comparison
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import grad
import torch.multiprocessing as mp
import h5py
import time
from functools import partial

# Add FluxNet source path
sys.path.insert(0, 'FluxNet/src')

# ==================== Configuration ====================
MODEL_PATHS = {
    '10dt': 'FluxNet/results/spinodal_decomposition/ablation_10dt/FluxNet_D_pf/best_model.pt',
    '100dt': 'FluxNet/results/spinodal_decomposition/ablation_100dt/FluxNet_D_pf/best_model.pt',
    '1000dt': 'FluxNet/results/spinodal_decomposition/ablation_1000dt/FluxNet_D_pf/best_model.pt',
}

MODEL_CONFIGS = {
    '10dt': {'base_channels': 32, 'num_blocks': 4, 'kernel_size': 3, 'neighborhood_size': 3},
    '100dt': {'base_channels': 32, 'num_blocks': 4, 'kernel_size': 5, 'neighborhood_size': 5},
    '1000dt': {'base_channels': 32, 'num_blocks': 6, 'kernel_size': 7, 'neighborhood_size': 9},
}

# Analysis parameters
THRESHOLD = 0.01  # 1% threshold for effective RF
NUM_SAMPLE_POINTS = 500  # 100 sample points as requested
IMAGE_SIZE = 128  # Size of test image
RANDOM_SEED = 666

# Output directory
OUTPUT_DIR = 'FluxNet/results/spinodal_decomposition/analysis_erf'

# Available GPUs
AVAILABLE_GPUS = [0, 1, 2, 3]


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(checkpoint_path, model_config, device):
    """Load a FluxNet_D model to specified device"""
    from models.fluxnet_d_2d import FluxNet_D

    model = FluxNet_D(
        in_channels=1,
        base_channels=model_config['base_channels'],
        num_blocks=model_config['num_blocks'],
        kernel_size=model_config['kernel_size'],
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d,
        neighborhood_size=model_config['neighborhood_size']
    )
    model.to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    model.eval()
    return model


def compute_theoretical_receptive_field(num_blocks, kernel_size):
    """
    Compute theoretical receptive field size for FluxNet_D model.
    """
    receptive_field_size = kernel_size
    for _ in range(num_blocks):
        receptive_field_size += 2 * (kernel_size - 1)
    return receptive_field_size


def get_channel_names(num_neighbors, neighborhood_size):
    """
    Get names for all channels in raw_fluxes output.
    """
    radius = neighborhood_size // 2
    channel_names = []

    # Outflow percentage channel
    channel_names.append("outflow_percentage_sigmoid")

    # Outflow distribution channels (softmax)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i != 0 or j != 0:
                channel_names.append(f"outflow_dist_softmax_dy{i:+d}_dx{j:+d}")

    # Inflow percentage channel
    channel_names.append("inflow_percentage_sigmoid")

    # Inflow distribution channels (softmax)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i != 0 or j != 0:
                channel_names.append(f"inflow_dist_softmax_dy{i:+d}_dx{j:+d}")

    return channel_names


def compute_all_channels_gradient_for_point(model, input_tensor, output_point, device):
    """
    Compute gradients for ALL channels at a single output point with SINGLE forward pass.

    Returns:
        List of gradient magnitude arrays (numpy), one per channel
    """
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)

    # Single forward pass
    x = input_tensor
    features = model.first_conv(x)

    for main_path, fusion_conv in model.res_blocks:
        identity = features
        features = main_path(features)
        features = torch.cat([features, identity], dim=1)
        features = fusion_conv(features)

    raw_fluxes = model.flux_conv(features)

    y, x_coord = output_point
    total_channels = raw_fluxes.shape[1]

    # Compute gradients for all channels (reusing computation graph)
    gradients = []
    for channel_idx in range(total_channels):
        target_output = raw_fluxes[0, channel_idx, y, x_coord]
        input_gradient = grad(target_output, input_tensor, retain_graph=True)[0]
        grad_magnitude = torch.abs(input_gradient[0, 0]).detach().cpu().numpy()
        gradients.append(grad_magnitude)

    return gradients


def calculate_effective_rf_size(grad_map, threshold=0.01):
    """Calculate effective receptive field size."""
    center_val = grad_map.max()
    if center_val == 0:
        return 0
    thresh_val = center_val * threshold
    rf_size = np.sum(grad_map > thresh_val)
    return rf_size


def process_single_point(point_data):
    """
    Process a single sample point - compute gradients for all channels.
    This function is designed to be called in parallel.
    """
    (point_idx, y, x, model, input_tensor, device, theoretical_rf, rf_radius, h, w, threshold,
     total_channels) = point_data

    # Compute all channel gradients with single forward pass
    gradients = compute_all_channels_gradient_for_point(model, input_tensor, (y, x), device)

    # Process each channel's gradient
    rf_regions = []
    rf_sizes = []

    for channel_idx, grad_magnitude in enumerate(gradients):
        # Normalize gradient
        if grad_magnitude.max() != 0:
            grad_normalized = grad_magnitude / grad_magnitude.max()
        else:
            grad_normalized = grad_magnitude

        # Extract receptive field region
        y_min = max(0, y - rf_radius)
        y_max = min(h, y + rf_radius + 1)
        x_min = max(0, x - rf_radius)
        x_max = min(w, x + rf_radius + 1)

        rf_region = grad_normalized[y_min:y_max, x_min:x_max]

        # Place RF region in centered theoretical RF
        rf_centered = np.zeros((theoretical_rf, theoretical_rf), dtype=np.float32)

        y_offset = rf_radius - (y - y_min)
        x_offset = rf_radius - (x - x_min)

        y_rf_min = max(0, y_offset)
        y_rf_max = min(theoretical_rf, y_offset + rf_region.shape[0])
        x_rf_min = max(0, x_offset)
        x_rf_max = min(theoretical_rf, x_offset + rf_region.shape[1])

        y_reg_min = max(0, -y_offset)
        y_reg_max = y_reg_min + (y_rf_max - y_rf_min)
        x_reg_min = max(0, -x_offset)
        x_reg_max = x_reg_min + (x_rf_max - x_rf_min)

        rf_centered[y_rf_min:y_rf_max, x_rf_min:x_rf_max] = rf_region[y_reg_min:y_reg_max, x_reg_min:x_reg_max]

        rf_regions.append(rf_centered)

        # Calculate effective RF size
        rf_size = calculate_effective_rf_size(rf_centered, threshold)
        rf_sizes.append(rf_size)

    return point_idx, rf_regions, rf_sizes


def process_model_on_gpu(gpu_id, model_name, model_path, model_config,
                         test_image, points, result_queue):
    """
    Process a single model on a specified GPU.
    """
    try:
        start_time = time.time()
        device = torch.device(f'cuda:{gpu_id}')

        print(f"[GPU {gpu_id}] Starting {model_name} processing...")

        # Load model
        model = load_model(model_path, model_config, device)
        model.eval()

        # Get model parameters
        num_blocks = model.num_blocks
        kernel_size = model.res_blocks[0][0].conv[1].kernel_size[0]
        total_channels = model.total_channels
        num_neighbors = model.num_neighbors
        neighborhood_size = model.neighborhood_size

        # Compute theoretical RF
        theoretical_rf = compute_theoretical_receptive_field(num_blocks, kernel_size)
        rf_radius = theoretical_rf // 2

        print(f"[GPU {gpu_id}] {model_name}: Theoretical RF = {theoretical_rf}x{theoretical_rf}, "
              f"Total channels = {total_channels}")

        # Prepare input tensor
        input_tensor = torch.from_numpy(test_image).float().unsqueeze(0).unsqueeze(0)
        h, w = test_image.shape

        # Get channel names
        channel_names = get_channel_names(num_neighbors, neighborhood_size)

        # Initialize accumulators
        accumulated_rfs = np.zeros((total_channels, theoretical_rf, theoretical_rf), dtype=np.float64)
        all_rf_sizes = np.zeros((total_channels, len(points)), dtype=np.float64)

        # Process all points
        for point_idx, (y, x) in enumerate(points):
            # Compute all channel gradients with single forward pass
            gradients = compute_all_channels_gradient_for_point(model, input_tensor, (y, x), device)

            for channel_idx, grad_magnitude in enumerate(gradients):
                # Normalize gradient
                if grad_magnitude.max() != 0:
                    grad_normalized = grad_magnitude / grad_magnitude.max()
                else:
                    grad_normalized = grad_magnitude

                # Extract receptive field region
                y_min = max(0, y - rf_radius)
                y_max = min(h, y + rf_radius + 1)
                x_min = max(0, x - rf_radius)
                x_max = min(w, x + rf_radius + 1)

                rf_region = grad_normalized[y_min:y_max, x_min:x_max]

                # Place RF region in centered theoretical RF
                rf_centered = np.zeros((theoretical_rf, theoretical_rf), dtype=np.float64)

                y_offset = rf_radius - (y - y_min)
                x_offset = rf_radius - (x - x_min)

                y_rf_min = max(0, y_offset)
                y_rf_max = min(theoretical_rf, y_offset + rf_region.shape[0])
                x_rf_min = max(0, x_offset)
                x_rf_max = min(theoretical_rf, x_offset + rf_region.shape[1])

                y_reg_min = max(0, -y_offset)
                y_reg_max = y_reg_min + (y_rf_max - y_rf_min)
                x_reg_min = max(0, -x_offset)
                x_reg_max = x_reg_min + (x_rf_max - x_rf_min)

                rf_centered[y_rf_min:y_rf_max, x_rf_min:x_rf_max] = \
                    rf_region[y_reg_min:y_reg_max, x_reg_min:x_reg_max]

                accumulated_rfs[channel_idx] += rf_centered

                # Calculate effective RF size
                rf_size = calculate_effective_rf_size(rf_centered, THRESHOLD)
                all_rf_sizes[channel_idx, point_idx] = rf_size

            # Progress report
            if (point_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (point_idx + 1) * (len(points) - point_idx - 1)
                print(f"[GPU {gpu_id}] {model_name}: {point_idx + 1}/{len(points)} points, "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

        # Compute final results for each channel
        channel_results = []
        for channel_idx in range(total_channels):
            avg_rf = accumulated_rfs[channel_idx] / len(points)
            avg_rf_normalized = avg_rf / avg_rf.max() if avg_rf.max() > 0 else avg_rf

            avg_rf_size = np.mean(all_rf_sizes[channel_idx])
            ratio = avg_rf_size / (theoretical_rf * theoretical_rf) * 100
            avg_rf_size_sqrt = np.sqrt(avg_rf_size)

            channel_results.append({
                'name': channel_names[channel_idx],
                'avg_rf': avg_rf_normalized,
                'avg_rf_size': avg_rf_size_sqrt,
                'ratio': ratio
            })

        elapsed_total = time.time() - start_time
        print(f"[GPU {gpu_id}] {model_name}: Completed in {elapsed_total:.1f}s")

        result_queue.put({
            'model_name': model_name,
            'theoretical_rf': theoretical_rf,
            'total_channels': total_channels,
            'num_neighbors': num_neighbors,
            'channel_results': channel_results,
            'success': True
        })

    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] Error processing {model_name}: {e}")
        traceback.print_exc()
        result_queue.put({
            'model_name': model_name,
            'success': False,
            'error': str(e)
        })


def save_model_results(model_name, theoretical_rf, total_channels, num_neighbors, channel_results):
    """
    Save results for a single model - EXACTLY matching original format.
    """
    # Create output directory for this model
    model_output_dir = os.path.join(OUTPUT_DIR, f'erf_results_{model_name}')
    os.makedirs(model_output_dir, exist_ok=True)

    # Save NPY file for each channel
    for channel_idx, result in enumerate(channel_results):
        safe_channel_name = result['name'].replace('(', '_').replace(')', '_').replace(',', '_')
        npy_path = os.path.join(model_output_dir,
                                f"{model_name}_rf_ch{channel_idx:02d}_{safe_channel_name}.npy")
        np.save(npy_path, result['avg_rf'])

    # Calculate branch averages
    # Outflow branch: channel 0 (sigmoid) + channels 1 to num_neighbors (softmax)
    outflow_results = channel_results[0:num_neighbors + 1]
    outflow_avg_size = np.mean([r['avg_rf_size'] for r in outflow_results])

    # Inflow branch: channel num_neighbors+1 (sigmoid) + remaining channels (softmax)
    inflow_results = channel_results[num_neighbors + 1:]
    inflow_avg_size = np.mean([r['avg_rf_size'] for r in inflow_results])

    # Overall average
    all_avg_size = np.mean([r['avg_rf_size'] for r in channel_results])
    all_avg_ratio = np.mean([r['ratio'] for r in channel_results])

    # Save statistics text file - EXACTLY matching original format
    stats_file = os.path.join(model_output_dir, f"{model_name}_rf_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Receptive Field Analysis for {model_name}\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Theoretical RF Size: {theoretical_rf}x{theoretical_rf}\n")
        f.write(f"Number of Channels: {total_channels}\n")
        f.write(f"Number of Sample Points: {NUM_SAMPLE_POINTS}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")

        f.write("Individual Channel Results:\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Ch':<4} {'Channel Name':<50} {'Effective RF':>15} {'Ratio':>10}\n")
        f.write("-" * 100 + "\n")

        for i, result in enumerate(channel_results):
            f.write(f"{i:02d}   {result['name']:<50} "
                    f"{result['avg_rf_size']:>6.2f}x{result['avg_rf_size']:<6.2f} "
                    f"{result['ratio']:>9.2f}%\n")

        f.write("=" * 100 + "\n\n")
        f.write(f"Branch Averages:\n")
        f.write(f"  Outflow branch average ERF: {outflow_avg_size:.2f}x{outflow_avg_size:.2f}\n")
        f.write(f"  Inflow branch average ERF: {inflow_avg_size:.2f}x{inflow_avg_size:.2f}\n")
        f.write(f"\nAll Channels Average:\n")
        f.write(f"  Effective RF Size: {all_avg_size:.2f}x{all_avg_size:.2f}\n")
        f.write(f"  Ratio: {all_avg_ratio:.2f}%\n")

    print(f"Results saved to: {model_output_dir}")

    return {
        'model_name': model_name,
        'theoretical_rf': theoretical_rf,
        'outflow_avg_erf': outflow_avg_size,
        'inflow_avg_erf': inflow_avg_size,
        'all_avg_erf': all_avg_size,
        'all_avg_ratio': all_avg_ratio,
        'channel_results': channel_results
    }


def generate_markdown_table(all_stats):
    """Generate markdown table with RF comparison - EXACTLY matching original format"""
    md_content = """# Effective Receptive Field Analysis Results

## Summary Table

| Model | Theoretical RF | Outflow Branch ERF | Inflow Branch ERF | Average ERF | ERF/TRF Ratio |
|-------|----------------|--------------------|--------------------|-------------|---------------|
"""

    for model_name in ['10dt', '100dt', '1000dt']:
        if model_name in all_stats:
            stats = all_stats[model_name]
            md_content += f"| FluxNet-D ({model_name}) | {stats['theoretical_rf']}×{stats['theoretical_rf']} | "
            md_content += f"{stats['outflow_avg_erf']:.2f}×{stats['outflow_avg_erf']:.2f} | "
            md_content += f"{stats['inflow_avg_erf']:.2f}×{stats['inflow_avg_erf']:.2f} | "
            md_content += f"{stats['all_avg_erf']:.2f}×{stats['all_avg_erf']:.2f} | "
            md_content += f"{stats['all_avg_ratio']:.2f}% |\n"

    md_content += """
## Notes

- **Theoretical RF (TRF)**: Calculated based on model architecture (kernel sizes and number of layers)
- **Effective RF (ERF)**: Measured empirically using gradient-based analysis with 1% threshold
- **ERF/TRF Ratio**: Percentage of theoretical receptive field that is effectively utilized

## Analysis Parameters

- Sample points: 100
- Threshold: 1% of maximum gradient magnitude
- Image size: 256×256
- Random seed: 666

"""

    return md_content


def load_test_image():
    """Load test image from H5 file"""
    test_image_index = 20
    model_evaluation_H5_folder = "FluxNet/dataset/spinodal_decomposition/test"

    for h5_file in os.listdir(model_evaluation_H5_folder):
        if h5_file.endswith(".h5"):
            h5_file_path = os.path.join(model_evaluation_H5_folder, h5_file)
            with h5py.File(h5_file_path, 'r') as f:
                phi_data = f['phi_data'][:]
                test_image = phi_data[test_image_index]
            break

    return test_image


def main():
    """Main function with multi-GPU parallel processing"""
    print("=" * 80)
    print("Optimized Effective Receptive Field Analysis for FluxNet_D Models")
    print("Using Multi-GPU Parallel Processing")
    print("=" * 80)

    total_start_time = time.time()

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Setup
    setup_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test image once
    print("\nLoading test image...")
    test_image = load_test_image()
    h, w = test_image.shape
    print(f"Test image size: {h}x{w}")

    # Generate sample points with fixed seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Compute max RF radius across all models for margin calculation
    max_theoretical_rf = 0
    for model_config in MODEL_CONFIGS.values():
        theoretical_rf = compute_theoretical_receptive_field(
            model_config['num_blocks'],
            model_config['kernel_size']
        )
        max_theoretical_rf = max(max_theoretical_rf, theoretical_rf)

    margin = max(5, max_theoretical_rf // 2)
    points = []
    for _ in range(NUM_SAMPLE_POINTS):
        y = random.randint(margin, h - margin - 1)
        x = random.randint(margin, w - margin - 1)
        points.append((y, x))

    print(f"Generated {NUM_SAMPLE_POINTS} sample points with margin={margin}")

    # Create result queue
    result_queue = mp.Queue()

    # Assign GPUs to models
    model_names = ['10dt', '100dt', '1000dt']
    gpu_assignments = {
        '10dt': AVAILABLE_GPUS[0],
        '100dt': AVAILABLE_GPUS[1],
        '1000dt': AVAILABLE_GPUS[2]
    }

    print("\nGPU Assignments:")
    for model_name, gpu_id in gpu_assignments.items():
        print(f"  {model_name} -> GPU {gpu_id}")

    # Start processes for each model
    processes = []
    for model_name in model_names:
        gpu_id = gpu_assignments[model_name]
        model_path = MODEL_PATHS[model_name]
        model_config = MODEL_CONFIGS[model_name]

        p = mp.Process(
            target=process_model_on_gpu,
            args=(gpu_id, model_name, model_path, model_config,
                  test_image, points, result_queue)
        )
        processes.append(p)
        p.start()
        print(f"\nStarted process for {model_name} on GPU {gpu_id}")

    # Collect results
    print("\n" + "=" * 80)
    print("Waiting for results...")
    print("=" * 80)

    all_stats = {}
    completed = 0

    while completed < len(model_names):
        result = result_queue.get()
        completed += 1

        if result['success']:
            model_name = result['model_name']
            print(f"\n[{completed}/{len(model_names)}] Received results for {model_name}")

            # Save results
            stats = save_model_results(
                model_name=result['model_name'],
                theoretical_rf=result['theoretical_rf'],
                total_channels=result['total_channels'],
                num_neighbors=result['num_neighbors'],
                channel_results=result['channel_results']
            )
            all_stats[model_name] = stats

            # Print summary for this model
            print(f"  Theoretical RF: {stats['theoretical_rf']}x{stats['theoretical_rf']}")
            print(f"  Outflow Branch ERF: {stats['outflow_avg_erf']:.2f}x{stats['outflow_avg_erf']:.2f}")
            print(f"  Inflow Branch ERF: {stats['inflow_avg_erf']:.2f}x{stats['inflow_avg_erf']:.2f}")
            print(f"  Average ERF: {stats['all_avg_erf']:.2f}x{stats['all_avg_erf']:.2f}")
            print(f"  ERF/TRF Ratio: {stats['all_avg_ratio']:.2f}%")
        else:
            print(
                f"\n[{completed}/{len(model_names)}] ERROR for {result['model_name']}: {result.get('error', 'Unknown error')}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Generate and save markdown table
    print("\n" + "=" * 80)
    print("Generating summary markdown...")

    md_content = generate_markdown_table(all_stats)
    md_path = os.path.join(OUTPUT_DIR, 'erf_analysis_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    total_elapsed = time.time() - total_start_time

    print(f"\n{'=' * 80}")
    print(f"Analysis complete!")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    print(f"Summary saved to: {md_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
