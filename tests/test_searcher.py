import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

from mergenetic.optimization import MergingProblem  # For type hinting and mocking
from mergenetic.searcher import Searcher


# Mock Algorithm class from pymoo
class MockAlgorithm:
    pass


class TestSearcher(unittest.TestCase):

    def setUp(self):
        self.mock_problem = MagicMock(spec=MergingProblem)
        self.mock_problem.discrete = False  # Default, can be overridden in tests
        self.mock_algorithm = MockAlgorithm()

        # Create a real temporary directory for results_path
        self.temp_results_dir = Path(
            tempfile.mkdtemp(prefix="mergenetic_test_searcher_")
        )

        self.n_iter = 10
        self.seed = 42
        self.run_id = "test_run_searcher_001"

        self.searcher = Searcher(
            problem=self.mock_problem,
            algorithm=self.mock_algorithm,
            results_path=str(self.temp_results_dir),
            n_iter=self.n_iter,
            seed=self.seed,
            run_id=self.run_id,
            verbose=False,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_results_dir)

    def test_initialization(self):
        self.assertEqual(self.searcher.problem, self.mock_problem)
        self.assertEqual(self.searcher.algorithm, self.mock_algorithm)
        self.assertEqual(self.searcher.results_path, self.temp_results_dir)
        self.assertEqual(self.searcher.n_iter, self.n_iter)
        self.assertEqual(self.searcher.seed, self.seed)
        self.assertEqual(self.searcher.run_id, self.run_id)
        self.assertFalse(self.searcher.verbose)

    @patch("mergenetic.searcher.searcher.minimize")
    def test_search_saves_dataframe_results(self, mock_minimize):
        mock_result = MagicMock()
        mock_result.X = np.array([0.1, 0.2, 0.3]) * 10
        mock_result.F = np.array([-0.9])
        mock_minimize.return_value = mock_result

        self.mock_problem.discrete = True
        # Make results_df a MagicMock so its to_csv can be asserted
        mock_df_to_save = MagicMock(spec=pd.DataFrame)
        self.mock_problem.results_df = mock_df_to_save

        returned_df = self.searcher.search()

        mock_minimize.assert_called_once_with(
            self.mock_problem,
            self.mock_algorithm,
            ("n_iter", self.n_iter),
            seed=self.seed,
            verbose=False,
        )
        np.testing.assert_array_almost_equal(
            self.searcher.result_X, np.array([0.1, 0.2, 0.3])
        )
        np.testing.assert_array_equal(self.searcher.result_F, np.array([-0.9]))

        expected_path = self.temp_results_dir / f"{self.run_id}.csv"
        mock_df_to_save.to_csv.assert_called_once_with(expected_path)  # Now this works
        self.assertEqual(returned_df, mock_df_to_save)  # Compare with the mock

    @patch("mergenetic.searcher.searcher.minimize")
    def test_search_saves_dict_results(self, mock_minimize):
        mock_result = MagicMock()
        mock_result.X = np.array([0.4, 0.5])
        mock_result.F = np.array([-0.8])
        mock_minimize.return_value = mock_result

        self.mock_problem.discrete = False
        # Make df1 and df2 MagicMocks
        df1_mock = MagicMock(spec=pd.DataFrame)
        df2_mock = MagicMock(spec=pd.DataFrame)
        self.mock_problem.results_df = {"key1": df1_mock, "key2": df2_mock}

        returned_dict = self.searcher.search()

        np.testing.assert_array_equal(self.searcher.result_X, np.array([0.4, 0.5]))

        expected_path1 = self.temp_results_dir / f"{self.run_id}_key1.csv"
        expected_path2 = self.temp_results_dir / f"{self.run_id}_key2.csv"

        df1_mock.to_csv.assert_called_once_with(expected_path1)
        df2_mock.to_csv.assert_called_once_with(expected_path2)
        self.assertEqual(returned_dict, self.mock_problem.results_df)

    @patch("mergenetic.searcher.searcher.minimize")
    @patch("mergenetic.searcher.searcher.logger")
    def test_search_no_results_df_attribute(self, mock_logger, mock_minimize):
        mock_minimize.return_value = MagicMock(X=np.array([1]), F=np.array([-1]))
        del self.mock_problem.results_df  # Ensure attribute doesn't exist

        result = self.searcher.search()
        self.assertIsNone(result)
        mock_logger.info.assert_any_call("Problem does not have 'results_df' to save.")

    @patch("mergenetic.searcher.searcher.minimize")
    @patch("mergenetic.searcher.searcher.logger")
    def test_search_results_df_wrong_type(self, mock_logger, mock_minimize):
        mock_minimize.return_value = MagicMock(X=np.array([1]), F=np.array([-1]))
        self.mock_problem.results_df = "not a dataframe or dict"

        result = self.searcher.search()
        self.assertIsNone(result)
        mock_logger.error.assert_called_once_with(
            "Problem 'results_df' is not a DataFrame or a dictionary."
        )

    def test_test_single_solution_dataframe_results(self):
        self.searcher.result_X = np.array([0.1, 0.2])
        self.mock_problem.test.return_value = (np.array([-0.95]), "Test description")

        # Make mock_results_data_df a MagicMock
        mock_results_data_df = MagicMock(spec=pd.DataFrame)
        self.mock_problem.get_data.return_value = mock_results_data_df
        self.mock_problem.results_df = mock_results_data_df

        self.searcher.test()

        self.mock_problem.test.assert_called_once()  # Check it was called
        # Manually check arguments for np.array comparison
        called_args, _ = self.mock_problem.test.call_args
        np.testing.assert_array_equal(called_args[0], self.searcher.result_X)

        self.mock_problem.get_data.assert_called_once()

        expected_path = self.temp_results_dir / f"{self.run_id}_test.csv"
        mock_results_data_df.to_csv.assert_called_once_with(expected_path)

    def test_test_single_solution_dict_results(self):
        self.searcher.result_X = np.array([0.3, 0.4])
        self.mock_problem.test.return_value = (np.array([-0.88]), "Test description")

        # Make df_test1 and df_test2 MagicMocks
        df_test1_mock = MagicMock(spec=pd.DataFrame)
        df_test2_mock = MagicMock(spec=pd.DataFrame)
        mock_results_data_dict = {
            "test_key1": df_test1_mock,
            "test_key2": df_test2_mock,
        }
        self.mock_problem.get_data.return_value = mock_results_data_dict
        self.mock_problem.results_df = {}

        self.searcher.test()

        self.mock_problem.test.assert_called_once()
        called_args, _ = self.mock_problem.test.call_args
        np.testing.assert_array_equal(called_args[0], self.searcher.result_X)

        self.mock_problem.get_data.assert_called_once()

        expected_path1 = self.temp_results_dir / f"{self.run_id}_test_key1_test.csv"
        expected_path2 = self.temp_results_dir / f"{self.run_id}_test_key2_test.csv"

        df_test1_mock.to_csv.assert_called_once_with(expected_path1)
        df_test2_mock.to_csv.assert_called_once_with(expected_path2)

    def test_test_multiple_solutions_dataframe_results(self):
        self.searcher.result_X = np.array([[0.1, 0.2], [0.3, 0.4]])

        self.mock_problem.test.side_effect = [
            (np.array([-0.9]), "Desc1"),
            (np.array([-0.8]), "Desc2"),
        ]

        # Make these MagicMocks
        mock_results_data_df1 = MagicMock(spec=pd.DataFrame)
        mock_results_data_df2 = MagicMock(spec=pd.DataFrame)
        self.mock_problem.get_data.side_effect = [
            mock_results_data_df1,
            mock_results_data_df2,
        ]
        self.mock_problem.results_df = MagicMock(
            spec=pd.DataFrame
        )  # Main results_df also a mock

        self.searcher.test()

        self.assertEqual(self.mock_problem.test.call_count, 2)

        # Manually check calls to problem.test with np.array
        calls = self.mock_problem.test.call_args_list
        np.testing.assert_array_equal(calls[0][0][0], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(calls[1][0][0], np.array([0.3, 0.4]))

        self.assertEqual(self.mock_problem.get_data.call_count, 2)

        expected_path = self.temp_results_dir / f"{self.run_id}_test.csv"

        mock_results_data_df1.to_csv.assert_any_call(expected_path)
        mock_results_data_df2.to_csv.assert_any_call(expected_path)
        # Check the last call to ensure it was with the second df's mock
        self.assertEqual(
            mock_results_data_df2.to_csv.call_args_list[-1], call(expected_path)
        )

    @patch("mergenetic.searcher.searcher.plt")
    def test_visualize_results_success(self, mock_plt):
        self.mock_problem.results_df = pd.DataFrame(
            {
                "step": [1, 2, 3],
                "objective_1": [0.1, 0.2, 0.3],
                "objective_2": [0.4, 0.3, 0.2],
                "phenotype_A": [10, 12, 11],
                "other_col": [1, 1, 1],
            }
        )

        self.searcher.visualize_results()

        self.assertEqual(mock_plt.figure.call_count, 3)  # 2 objectives + 1 phenotype
        self.assertEqual(mock_plt.plot.call_count, 3)
        self.assertEqual(mock_plt.title.call_count, 3)
        self.assertEqual(mock_plt.xlabel.call_count, 3)
        self.assertEqual(mock_plt.ylabel.call_count, 3)
        self.assertEqual(mock_plt.grid.call_count, 3)
        self.assertEqual(mock_plt.show.call_count, 3)

        # Check one of the plot calls for detail (e.g., first objective)
        # This can be more detailed if specific plot contents need verification
        mock_plt.title.assert_any_call("Metric: objective_1")
        mock_plt.ylabel.assert_any_call("objective_1")

        mock_plt.title.assert_any_call("Metric: objective_2")
        mock_plt.ylabel.assert_any_call("objective_2")

        mock_plt.title.assert_any_call("Phenotype: phenotype_A")
        mock_plt.ylabel.assert_any_call("phenotype_A")

    def test_visualize_results_no_results_df(self):
        del self.mock_problem.results_df  # Ensure attribute doesn't exist
        with self.assertRaisesRegex(
            AttributeError, "Problem does not have 'results_df' to visualize results."
        ):
            self.searcher.visualize_results()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
