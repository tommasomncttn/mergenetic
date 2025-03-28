import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from mergenetic.estimator.utils import estimate_fitness

# ==========================
#  DATA CLASSES FOR ESTIMATION
# ==========================

@dataclass
class PerformanceEstimationParameters:
    """Stores parameters for accuracy estimation."""
    thetas: List[np.array]
    sample_weights: np.array
    sample_ids: np.array
    bench: str
    mode: str   # "mean", "weighted", "mpirt", "gmpirt"

# ==========================
#  PERFORMANCE ESTIMATOR
# ==========================

class PerformanceEstimator:
    """
    Evaluates the accuracy of a model using anchored evaluation techniques.
    """

    est_parameters: PerformanceEstimationParameters

    def __init__(self, est_parameters: PerformanceEstimationParameters) -> None:
        """
        Initializes the performance estimator.

        Parameters
        ----------
        est_parameters : PerformanceEstimationParameters
            Parameters for accuracy estimation.
        """
        self.est_parameters = est_parameters

    def __str__(self) -> str:
        return "Performance Estimator for LLM predictions against mathematical word problems."

    def estimate_accuracy(self, correctness: pd.Series) -> float:
        """
        Estimates the accuracy of the model based on the correctness of the predictions.

        Parameters
        ----------
        correctness : pd.Series
            Series containing the correctness of the predictions.

        Returns
        -------
        float
            Estimated accuracy of the model.
        """
        # Convert boolean results to integers (1 = correct, 0 = incorrect)
        y = correctness.astype(int).values

        match self.est_parameters.mode:
            case 'mean':
                return y.mean()
            case 'mpirt' | 'gmpirt':
                if self.est_parameters.mode == 'gmpirt' and len(self.est_parameters.thetas) == 1:
                    raise ValueError("GMPIRT requires multiple thetas.")
                
                # Estimate accuracy using anchoring methods
                estimates = estimate_fitness(
                    y, self.est_parameters.thetas, self.est_parameters.bench,
                    self.est_parameters.sample_ids, self.est_parameters.sample_weights
                )
                return estimates[self.est_parameters.mode]
            case 'weighted':
                return (self.est_parameters.sample_weights * y).sum()
            
        raise ValueError(f"Invalid mode '{self.est_parameters.mode}' for accuracy estimation.")