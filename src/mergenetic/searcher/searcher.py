from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize

from mergenetic.optimization import MergingProblem

logger = getLogger(__name__)


class Searcher:
    def __init__(
        self,
        problem: MergingProblem,
        algorithm: Algorithm,
        results_path: str,
        n_iter: int,
        seed: int,
        run_id: str,
        verbose: bool = True,
    ):
        """
        Initializes the Seaercher with a problem, an algorithm, and a path to save results.

        :param problem: An instance of a Problem subclass.
        :param algorithm: An instance of an Algorithm class.
        :param results_path: Path to save the optimization results.
        :param n_iter: Number of iterations to run the optimization algorithm.
        :param seed: Random seed for reproducibility.
        :param verbose: Whether to print optimization progress
        """
        self.problem = problem
        self.algorithm = algorithm
        self.results_path = Path(results_path)
        self.n_iter = n_iter
        self.seed = seed
        self.verbose = verbose
        self.run_id = run_id

    def search(self) -> pd.DataFrame:
        """Executes the search algorithm and saves the results to a CSV file."""
        result = minimize(
            self.problem,
            self.algorithm,
            ("n_iter", self.n_iter),
            seed=self.seed,
            verbose=self.verbose,
        )

        self.result_X = result.X / 10 if self.problem.discrete else result.X
        self.result_F = result.F
        logger.info(f"Best solution found: {result.X}. Best function value: {result.F}")

        if hasattr(self.problem, "results_df"):
            if isinstance(self.problem.results_df, pd.DataFrame):
                search_path = self.results_path / Path(f"{self.run_id}.csv")
                self.problem.results_df.to_csv(search_path)
                return self.problem.results_df
            elif isinstance(self.problem.results_df, dict):
                for key, value in self.problem.results_df.items():
                    search_path = self.results_path / Path(f"{self.run_id}_{key}.csv")
                    value.to_csv(search_path)
                return self.problem.results_df
            else:
                logger.error("Problem 'results_df' is not a DataFrame or a dictionary.")
                return None
        else:
            logger.info("Problem does not have 'results_df' to save.")
            return None

    def test(self):
        # check that a test dataframe was given
        # check that the path is valid
        logger.info("Starting the test...")

        if len(self.result_X.shape) > 1:
            for x in self.result_X:
                # check that is a numpy array
                assert type(x) == np.ndarray, "not a numpy array"
                fit, _ = self.problem.test(x)
                results_df = self.problem.get_data()
                if isinstance(self.problem.results_df, pd.DataFrame):
                    path = self.results_path / Path(f"{self.run_id}_test.csv")
                    logger.info(f"The genotype {x} got fitness {fit}")
                    results_df.to_csv(
                        self.results_path / Path(f"{self.run_id}_test.csv")
                    )
                else:
                    for key, value in results_df.items():
                        path = self.results_path / Path(f"{self.run_id}_test_{key}.csv")
                        logger.info(f"The genotype {x} got fitness {fit} on {key}.")
                        value.to_csv(path)
            logger.info("TEST MODE COMPLETED")
        else:
            # check that is a numpy array
            assert type(self.result_X) == np.ndarray, "not a numpy array"
            accuracy, _ = self.problem.test(self.result_X)
            logger.info(f"The genotype {self.result_X} got accuracy {accuracy}")
            results_df = self.problem.get_data()

            if isinstance(self.problem.results_df, pd.DataFrame):
                path = self.results_path / Path(f"{self.run_id}_test.csv")
                results_df.to_csv(path)
            else:
                for key, value in results_df.items():
                    path = self.results_path / Path(f"{self.run_id}_{key}_test.csv")
                    value.to_csv(path)
            logger.info("TEST MODE COMPLETED")

    def visualize_results(self) -> None:
        """Plots metrics and phenotypes over the optimization steps from the results_df."""
        if not hasattr(self.problem, "results_df"):
            raise AttributeError(
                "Problem does not have 'results_df' to visualize results."
            )

        df = self.problem.results_df
        metrics = [col for col in df.columns if "objective" in col]
        phenotypes = [col for col in df.columns if "phenotype" in col]

        for metric in metrics:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[metric], marker="o", linestyle="-")
            plt.title(f"Metric: {metric}")
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.grid(True)
            plt.show()

        for phenotype in phenotypes:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[phenotype], marker="x", linestyle="--")
            plt.title(f"Phenotype: {phenotype}")
            plt.xlabel("Step")
            plt.ylabel(phenotype)
            plt.grid(True)
            plt.show()
