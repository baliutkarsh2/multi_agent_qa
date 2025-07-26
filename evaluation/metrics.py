"""
Standard metrics used by the evaluation framework.
"""

from __future__ import annotations

from typing import List, Dict, Any
from statistics import mean


def success_rate(exec_reports: List[Dict[str, Any]]) -> float:
    successes = [r["report"]["success"] for r in exec_reports]
    return sum(successes) / max(len(successes), 1)


def avg_duration(exec_reports: List[Dict[str, Any]]) -> float:
    durations = [r["report"]["duration"] for r in exec_reports]
    return mean(durations) if durations else 0.0 