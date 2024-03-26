"""Composite Source Generator

Generate a composite harmonized dataset from multiple sources according to defined
source priorities and matching algorithms.
"""

from ._compose import compose
from ._models import PriorityDefinition, StrategyDefinition
from ._strategies.substitution import SubstitutionStrategy

__all__ = [
    "compose",
    "PriorityDefinition",
    "StrategyDefinition",
    "SubstitutionStrategy",
]
