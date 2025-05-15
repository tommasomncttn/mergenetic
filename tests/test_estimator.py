import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mergenetic.estimator.perf_estimation import (
    PerformanceEstimationParameters,
    PerformanceEstimator,
)


class TestPerformanceEstimator(unittest.TestCase):

    def setUp(self):
        # Common parameters for many tests
        self.dummy_thetas_multiple = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        self.dummy_thetas_single = [np.array([0.1, 0.2])]
        self.dummy_sample_weights = np.array([0.5, 0.5, 1.0, 1.0])
        self.dummy_sample_ids = np.array(
            [0, 1, 2, 3]
        )  # Corresponds to correctness series length
        self.dummy_bench = "gsm8k"
        self.correctness_series = pd.Series([True, False, True, True])  # 0.75 mean

    def test_init_and_str(self):
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="mean",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        self.assertEqual(estimator.est_parameters, params)
        self.assertEqual(
            str(estimator),
            "Performance Estimator for LLM predictions against mathematical word problems.",
        )

    def test_estimate_accuracy_mean_mode(self):
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,  # Not used in mean mode
            sample_weights=self.dummy_sample_weights,  # Not used
            sample_ids=self.dummy_sample_ids,  # Not used
            bench=self.dummy_bench,  # Not used
            mode="mean",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        accuracy = estimator.estimate_accuracy(self.correctness_series)
        self.assertEqual(accuracy, 0.75)

    def test_estimate_accuracy_weighted_mode(self):
        # correctness_series = [1, 0, 1, 1]
        # sample_weights     = [0.5, 0.5, 1.0, 1.0]
        # weighted_correct   = [0.5, 0.0, 1.0, 1.0]
        # sum = 2.5
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,  # Not used
            sample_weights=self.dummy_sample_weights[: len(self.correctness_series)],
            sample_ids=self.dummy_sample_ids,  # Not used
            bench=self.dummy_bench,  # Not used
            mode="weighted",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        accuracy = estimator.estimate_accuracy(self.correctness_series)
        expected_accuracy = (
            self.correctness_series.astype(int).values * params.sample_weights
        ).sum()
        self.assertEqual(accuracy, expected_accuracy)

    @patch("mergenetic.estimator.perf_estimation.estimate_fitness")
    def test_estimate_accuracy_mpirt_mode(self, mock_estimate_fitness):
        mock_estimate_fitness.return_value = {
            "mpirt": 0.65,
            "weighted_avg": 0.7,
            "gmpirt": 0.72,
        }
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_multiple,
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="mpirt",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        accuracy = estimator.estimate_accuracy(self.correctness_series)

        self.assertEqual(mock_estimate_fitness.call_count, 1)
        called_args, called_kwargs = mock_estimate_fitness.call_args
        np.testing.assert_array_equal(
            called_args[0], self.correctness_series.astype(int).values
        )
        self.assertEqual(
            called_args[1], params.thetas
        )  # List of arrays comparison is fine
        self.assertEqual(called_args[2], params.bench)
        np.testing.assert_array_equal(called_args[3], params.sample_ids)
        np.testing.assert_array_equal(called_args[4], params.sample_weights)

        self.assertEqual(accuracy, 0.65)

    @patch("mergenetic.estimator.perf_estimation.estimate_fitness")
    def test_estimate_accuracy_gmpirt_mode(self, mock_estimate_fitness):
        mock_estimate_fitness.return_value = {
            "mpirt": 0.65,
            "weighted_avg": 0.7,
            "gmpirt": 0.72,
        }
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_multiple,
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="gmpirt",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        accuracy = estimator.estimate_accuracy(self.correctness_series)

        self.assertEqual(mock_estimate_fitness.call_count, 1)
        called_args, called_kwargs = mock_estimate_fitness.call_args
        np.testing.assert_array_equal(
            called_args[0], self.correctness_series.astype(int).values
        )
        self.assertEqual(called_args[1], params.thetas)
        self.assertEqual(called_args[2], params.bench)
        np.testing.assert_array_equal(called_args[3], params.sample_ids)
        np.testing.assert_array_equal(called_args[4], params.sample_weights)

        self.assertEqual(accuracy, 0.72)

    def test_estimate_accuracy_gmpirt_mode_single_theta_raises_error(self):
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,  # Single theta
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="gmpirt",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        with self.assertRaisesRegex(ValueError, "GMPIRT requires multiple thetas."):
            estimator.estimate_accuracy(self.correctness_series)

    @patch("mergenetic.estimator.perf_estimation.estimate_fitness")
    def test_estimate_accuracy_pirt_gpirt_naming_with_single_theta(
        self, mock_estimate_fitness
    ):
        # estimate_fitness itself handles the naming pirt/gpirt based on len(thetas)
        # We are testing that PerformanceEstimator correctly extracts the value based on its mode
        # when estimate_fitness returns the pirt/gpirt keys.

        # Test PIRT mode
        mock_estimate_fitness.return_value = {
            "pirt": 0.60,
            "weighted_avg": 0.7,
            "gpirt": 0.68,
        }  # estimate_fitness returns 'pirt'
        params_pirt = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,  # Single theta
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="mpirt",  # PerformanceEstimator uses 'mpirt' which should map to 'pirt' if single theta
        )
        estimator_pirt = PerformanceEstimator(est_parameters=params_pirt)
        accuracy_pirt = estimator_pirt.estimate_accuracy(self.correctness_series)
        self.assertEqual(accuracy_pirt, 0.60)  # Should pick up 'pirt' value

        self.assertEqual(mock_estimate_fitness.call_count, 1)
        called_args, called_kwargs = mock_estimate_fitness.call_args
        np.testing.assert_array_equal(
            called_args[0], self.correctness_series.astype(int).values
        )
        self.assertEqual(called_args[1], params_pirt.thetas)
        self.assertEqual(called_args[2], params_pirt.bench)
        np.testing.assert_array_equal(called_args[3], params_pirt.sample_ids)
        np.testing.assert_array_equal(called_args[4], params_pirt.sample_weights)

        # Test GPIRT mode (PerformanceEstimator mode is 'gmpirt')
        # This case is already handled by the ValueError for GMPIRT with single theta.
        # If we wanted to test 'gpirt' key extraction specifically, we would need to ensure
        # the ValueError for single theta in 'gmpirt' mode is not raised first,
        # or test a different mode that might use 'gpirt'.
        # The fix in perf_estimation.py for 'gmpirt' mode with single theta already raises an error,
        # so we cannot reach the `estimates[mode_key]` part for 'gpirt' with a single theta
        # if the mode is 'gmpirt'.

        # The test for 'mpirt' mode with single theta (above) now correctly tests the 'pirt' key extraction.
        # The 'gmpirt' mode with single theta is correctly tested by
        # `test_estimate_accuracy_gmpirt_mode_single_theta_raises_error`.

    def test_estimate_accuracy_invalid_mode(self):
        params = PerformanceEstimationParameters(
            thetas=self.dummy_thetas_single,
            sample_weights=self.dummy_sample_weights,
            sample_ids=self.dummy_sample_ids,
            bench=self.dummy_bench,
            mode="invalid_mode",
        )
        estimator = PerformanceEstimator(est_parameters=params)
        with self.assertRaisesRegex(
            ValueError, "Invalid mode 'invalid_mode' for accuracy estimation."
        ):
            estimator.estimate_accuracy(self.correctness_series)


if __name__ == "__main__":
    unittest.main()
