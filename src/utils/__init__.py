"""
Utilities module exports
"""

from .visualization import (
    create_bwr_custom_colormap,
    create_wr_custom_colormap,
    model_performance_during_training,
    plot_train_loss_curve,
    model_performance_visualization,
    model_performance_visualization11,
    model_performance_visualization13,
    delta_phi_visualization,
    visualize_conservation,
    # 1D visualization
    plot_1d_prediction_comparison,
    plot_1d_spacetime_heatmap,
    plot_1d_error_curve,
    # Shallow water visualization
    plot_shallow_water_fields,
    save_field_to_tecplot,
    # Statistical curves
    plot_violation_curve,
    plot_rollout_error_vs_horizon,
    plot_pushforward_loss_breakdown,
)

__all__ = [
    'create_bwr_custom_colormap',
    'create_wr_custom_colormap',
    'model_performance_during_training',
    'plot_train_loss_curve',
    'model_performance_visualization',
    'model_performance_visualization11',
    'model_performance_visualization13',
    'delta_phi_visualization',
    'visualize_conservation',
    'plot_1d_prediction_comparison',
    'plot_1d_spacetime_heatmap',
    'plot_1d_error_curve',
    'plot_shallow_water_fields',
    'save_field_to_tecplot',
    'plot_violation_curve',
    'plot_rollout_error_vs_horizon',
    'plot_pushforward_loss_breakdown',
]
