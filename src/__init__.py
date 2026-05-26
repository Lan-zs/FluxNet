"""
FluxNet Package

主要模块:
- models: 所有 FluxNet 和 Baseline 模型
- training: 训练器、数据加载器、配置
- evaluation: 评估器、指标计算
- utils: 可视化工具
"""

__version__ = '1.0.0'

from . import models
from . import training
from . import evaluation
from . import utils

__all__ = ['models', 'training', 'evaluation', 'utils']
