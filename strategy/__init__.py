"""
Daily trading strategy package.
"""

from .daily_signal import (
    evaluate_signals,
    get_latest_signal,
)

__all__ = [
    "evaluate_signals",
    "get_latest_signal",
]