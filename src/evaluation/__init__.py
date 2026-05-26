"""
Evaluation Module
"""

from .evaluator_unified import (
    UnifiedEvaluator,
    EvaluationMetrics,
    evaluate_model_on_test_set
)

from .evaluator_sw import (
    ShallowWaterEvaluator,
    evaluate_shallow_water
)

__all__ = [
    'UnifiedEvaluator',
    'EvaluationMetrics',
    'evaluate_model_on_test_set',
    # Shallow water specific
    'ShallowWaterEvaluator',
    'evaluate_shallow_water',
]
