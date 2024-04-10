"""T-test for comparing two models.

Usage: python ttest.py <model dir path 1> <model dir path 2>
"""


import os
import sys
import pickle
import numpy as np
from scipy import stats
from typing import List, Dict
from RecSys.metrics import TestMetric


try:
    PATH1 = sys.argv[1]
    PATH2 = sys.argv[2]

    with open(os.path.join(PATH1, "test_results.pkl"), "rb") as f:
        metrics_1 = pickle.load(f)

    with open(os.path.join(PATH2, "test_results.pkl"), "rb") as f:
        metrics_2 = pickle.load(f)

except Exception:
    raise ValueError("Usage: python ttest.py <model dir path 1> <model dir path 2>")


def process_metrics(metrics: List[TestMetric]) -> Dict[str, np.ndarray]:
    """Process metrics.

    Args:
        metrics (List[MetricsPrint]): Metrics.

    Returns:
        Dict[str, float]: Processed metrics.
    """
    processed_metrics = dict()
    for metric in metrics:
        metric_name = metric.name
        metric_values = metric.metric.cpu().numpy()
        processed_metrics[metric_name] = metric_values
    return processed_metrics


def ttest(metrics_1: Dict[str, np.ndarray], metrics_2: Dict[str, np.ndarray]) -> None:
    """T-test for comparing two models.

    Args:
        metrics_1 (Dict[str, np.ndarray]): Metrics for model 1.
        metrics_2 (Dict[str, np.ndarray]): Metrics for model 2.
    """
    for metric_name, metric_values_1 in metrics_1.items():
        if metric_name not in metrics_2:
            continue
        metric_values_2 = metrics_2[metric_name]
        t, p = stats.ttest_ind(metric_values_1, metric_values_2)
        print(f"{metric_name}: t={t:.3f}, p={p:.3e}, mean_1={np.mean(metric_values_1):.4f}, mean_2={np.mean(metric_values_2):.4f}")


if __name__ == "__main__":
    metrics_1 = process_metrics(metrics_1)
    metrics_2 = process_metrics(metrics_2)
    ttest(metrics_1, metrics_2)
