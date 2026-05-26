"""
Training Configuration Classes
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class TrainingConfig:
    """
    Unified training configuration for all datasets and models

    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        ndt: Number of timesteps to skip (prediction horizon)
        loss_criterion: 'MSE' or 'MAE'

        # Loss weight management
        loss_weight_mode: 'adaptive' or 'manual'
            - 'adaptive': Auto-balance based on initial loss magnitudes
            - 'manual': Use user-specified weights
        loss_weights: Dict of manual weights (used when loss_weight_mode='manual')

        # Soft conservation (for baseline models only)
        soft_conservation_weight: Weight for soft conservation penalty (0.0 to disable)

        # Pushforward training
        use_pushforward: Whether to use pushforward training
        unroll_steps: Number of steps to unroll in pushforward
        unroll_weight: Weight for pushforward loss vs onestep loss

        # Scheduler
        scheduler_patience: Patience for ReduceLROnPlateau
        scheduler_factor: Factor to reduce LR

        # Data loading
        num_workers: Number of data loading workers

        # Saving
        save_interval: Save checkpoint every N epochs
    """

    # Basic training
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    ndt: int = 1
    loss_criterion: str = 'MSE'  # 'MSE' or 'MAE'

    # Loss weight management
    loss_weight_mode: str = 'adaptive'  # 'adaptive' or 'manual'
    loss_weights: Dict[str, float] = field(default_factory=dict)  # For manual mode

    # Soft conservation (for ablation with baseline models)
    soft_conservation_weight: float = 0.0  # 0.0 = disabled, only for baseline

    # DCL loss weight (for FluxNet-D dual consistency loss)
    dcl_weight: float = 0.1  # Weight for dual consistency loss in FluxNet-D

    # Pushforward training
    use_pushforward: bool = False
    unroll_steps: int = 5
    unroll_weight: float = 0.5

    # Scheduler
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5

    # Data loading
    num_workers: int = 4

    # Saving
    save_interval: int = 10

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'ndt': self.ndt,
            'loss_criterion': self.loss_criterion,
            'loss_weight_mode': self.loss_weight_mode,
            'loss_weights': self.loss_weights,
            'soft_conservation_weight': self.soft_conservation_weight,
            'dcl_weight': self.dcl_weight,
            'use_pushforward': self.use_pushforward,
            'unroll_steps': self.unroll_steps,
            'unroll_weight': self.unroll_weight,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
            'num_workers': self.num_workers,
            'save_interval': self.save_interval,
        }


@dataclass
class ModelConfig:
    """
    Model architecture configuration

    Args:
        model_type: Model class name
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks
        kernel_size: Convolutional kernel size
        neighborhood_size: Flux network neighborhood size
        lower_bound: Lower bound for L/D variants
        upper_bound: Upper bound for D variant
        head_config: Configuration for FluxNet_SW_2D ('NNN', 'LNN', 'LPP', 'LPP_h')
        prediction_mode: For baseline models ('direct' or 'residual')
        bound_mode: For CNN baseline ('none', 'lower', 'double')
        bound_h: For FluxNet_SW_Baseline (whether to bound h >= 0)

        # FNO specific
        modes: Number of Fourier modes (FNO)
        width: Hidden channel width (FNO)
        num_layers: Number of FNO layers
    """

    model_type: str
    base_channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 3
    neighborhood_size: int = 15

    # Bounds for L/D variants
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Shallow water FluxNet specific
    head_config: str = 'LNN'

    # Baseline specific
    prediction_mode: str = 'residual'
    bound_mode: str = 'none'  # 'none', 'lower', 'double' for CNN baseline
    bound_h: bool = False     # For FluxNet_SW_Baseline
    projection_mode: str = 'none'  # 'none', 'box', 'box_mass' for projection baselines

    # FNO specific
    modes: int = 16
    width: int = 64
    num_layers: int = 4

    def to_dict(self):
        """Convert to dictionary for model initialization"""
        # FNO models have different parameters
        if self.model_type == 'FNO_SW':
            return {
                'modes1': self.modes,
                'modes2': self.modes,
                'width': self.width,
                'num_layers': self.num_layers,
                'prediction_mode': self.prediction_mode,
            }

        # FNO_SW_Proj (FNO with projection for shallow water)
        if self.model_type == 'FNO_SW_Proj':
            return {
                'modes1': self.modes,
                'modes2': self.modes,
                'width': self.width,
                'num_layers': self.num_layers,
                'projection_mode': self.projection_mode,
                'prediction_mode': self.prediction_mode,
            }

        # FNO_FluxLAP (FNO backbone with LAP head for shallow water)
        if self.model_type == 'FNO_FluxLAP':
            return {
                'modes1': self.modes,
                'modes2': self.modes,
                'width': self.width,
                'num_layers': self.num_layers,
                'neighborhood_size': self.neighborhood_size,
                'lower_bound': self.lower_bound if self.lower_bound is not None else 0.0,
            }

        # # FluxNet_D_Dirichlet specific (same interface as FluxNet_D_1D)
        # if self.model_type == 'FluxNet_D_Dirichlet_1D':
        #     config['neighborhood_size'] = self.neighborhood_size
        #     if self.lower_bound is not None:
        #         config['lower_bound'] = self.lower_bound
        #     if self.upper_bound is not None:
        #         config['upper_bound'] = self.upper_bound
        #
        # # FluxGNN_D specific (same interface as FluxNet_D_1D)
        # if self.model_type == 'FluxGNN_D_1D':
        #     config['neighborhood_size'] = self.neighborhood_size
        #     if self.lower_bound is not None:
        #         config['lower_bound'] = self.lower_bound
        #     if self.upper_bound is not None:
        #         config['upper_bound'] = self.upper_bound

        # FluxGNN specific (same interface as FNO_1D)
        if self.model_type == 'FluxGNN_1D':
            return {
                'modes': self.modes,
                'width': self.width,
                'num_layers': self.num_layers,
                'prediction_mode': self.prediction_mode,
            }

        config = {
            'base_channels': self.base_channels,
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
        }

        # Add neighborhood_size for FluxNet models (not baseline)
        if 'FluxNet' in self.model_type and 'Baseline' not in self.model_type and 'SW' not in self.model_type:
            config['neighborhood_size'] = self.neighborhood_size

        # FluxGNN_D_1D also uses neighborhood_size (same interface as FluxNet_D_1D)
        if self.model_type == 'FluxGNN_D_1D':
            config['neighborhood_size'] = self.neighborhood_size
            if self.lower_bound is not None:
                config['lower_bound'] = self.lower_bound
            if self.upper_bound is not None:
                config['upper_bound'] = self.upper_bound

        # FluxNet_SW_2D specific
        if self.model_type == 'FluxNet_SW_2D':
            config['neighborhood_size'] = self.neighborhood_size
            config['head_config'] = self.head_config
            if self.lower_bound is not None:
                config['lower_bound'] = self.lower_bound

        # Add bounds if specified (for L/D variants)
        if self.lower_bound is not None and 'SW' not in self.model_type:
            config['lower_bound'] = self.lower_bound
        if self.upper_bound is not None:
            config['upper_bound'] = self.upper_bound

        # FluxNet_SW_Baseline specific
        if self.model_type == 'FluxNet_SW_Baseline':
            config['prediction_mode'] = self.prediction_mode
            config['bound_h'] = self.bound_h
            if self.lower_bound is not None:
                config['lower_bound'] = self.lower_bound

        # CNN_Baseline specific
        if 'CNN_Baseline' in self.model_type:
            config['prediction_mode'] = self.prediction_mode
            config['bound_mode'] = self.bound_mode
            if self.lower_bound is not None:
                config['lower_bound'] = self.lower_bound
            if self.upper_bound is not None:
                config['upper_bound'] = self.upper_bound

        return config

    def get_experiment_name(self) -> str:
        """Generate experiment name from config"""
        parts = [self.model_type]
        parts.append(f"c{self.base_channels}")
        parts.append(f"b{self.num_blocks}")
        parts.append(f"k{self.kernel_size}")

        if 'FluxNet' in self.model_type and 'Baseline' not in self.model_type:
            parts.append(f"n{self.neighborhood_size}")

        if self.model_type == 'FluxGNN_D_1D':
            parts.append(f"n{self.neighborhood_size}")

        if self.lower_bound is not None:
            parts.append(f"lb{self.lower_bound}")
        if self.upper_bound is not None:
            parts.append(f"ub{self.upper_bound}")

        if self.model_type == 'FluxNet_SW_2D':
            parts.append(self.head_config)

        if 'Baseline' in self.model_type:
            parts.append(self.prediction_mode)

        return "_".join(parts)
