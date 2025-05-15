import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mergenetic.optimization.predefined_problems import (
    ConfigLmEvalMultiObjectivePE,
    ConfigMultiLingualPE,
    ConfigPE,
    CrossLingualMathProblem,
    LmEvalMultiObjectiveProblem,
    MultilingualMergingProblem,
)


class TestConfigClasses(unittest.TestCase):
    def test_config_pe_instantiation(self):
        config = ConfigPE(
            thetas=[[0.1, 0.2]],
            sample_ids=[1, 2, 3],
            weights=[0.5, 0.5, 0.5],
            correct_metric="accuracy",
            bench="gsm8k",
            mode="mean",
        )
        self.assertEqual(config.thetas, [[0.1, 0.2]])
        self.assertEqual(config.correct_metric, "accuracy")
        self.assertIsNotNone(config)

    def test_config_multilingual_pe_instantiation(self):
        config = ConfigMultiLingualPE(
            sample_ids={"en": [1], "fr": [2]},
            weights={"en": [0.5], "fr": [0.5]},
            thetas=[[0.1]],
            bench="mmlu",
            mode="mpirt",
            correct_metric="bleu",
        )
        self.assertEqual(config.sample_ids["en"], [1])
        self.assertEqual(config.correct_metric, "bleu")
        self.assertIsNotNone(config)

    def test_config_lm_eval_multiobjective_pe_instantiation(self):
        config = ConfigLmEvalMultiObjectivePE(
            tasks=["task1", "task2"],
            correct_metric="exact_match",
            sample_ids={"task1": [1, 2], "task2": [3, 4]},
            additional_templates_folder="/tmp/templates",
        )
        self.assertEqual(config.tasks, ["task1", "task2"])
        self.assertEqual(config.additional_templates_folder, "/tmp/templates")
        self.assertIsNotNone(config)

    def test_config_lm_eval_multiobjective_pe_post_init_validation(self):
        with self.assertRaisesRegex(
            ValueError, "Tasks should be a list of task names."
        ):
            ConfigLmEvalMultiObjectivePE(
                tasks="not_a_list", correct_metric="acc", sample_ids={}
            )

        with self.assertRaisesRegex(
            ValueError, "Sample IDs should be a dictionary of lists of integers."
        ):
            ConfigLmEvalMultiObjectivePE(
                tasks=["t1"], correct_metric="acc", sample_ids="not_a_dict"
            )

        with self.assertRaisesRegex(
            ValueError, "Correctness metric should be a string."
        ):
            ConfigLmEvalMultiObjectivePE(
                tasks=["t1"], correct_metric=123, sample_ids={"t1": [1]}
            )

        with self.assertRaisesRegex(
            ValueError, "Additional templates folder should be a string or None."
        ):
            ConfigLmEvalMultiObjectivePE(
                tasks=["t1"],
                correct_metric="acc",
                sample_ids={"t1": [1]},
                additional_templates_folder=123,
            )


class TestCrossLingualMathProblem(unittest.TestCase):
    def setUp(self):
        self.mock_merger = MagicMock(name="MergerInstance")
        self.search_df = pd.DataFrame({"prompt": ["q1"], "answer": ["a1"]})
        self.test_df = pd.DataFrame({"prompt": ["q_test1"], "answer": ["a_test1"]})
        self.lang_id = "it"
        self.conf_pe = ConfigPE(
            thetas=[[0.1]],
            sample_ids=[0],
            weights=[1.0],
            correct_metric="exact_match",
            mode="mean",
        )
        self.lm_eval_tasks = {
            "search": {"it": "arc_it_search"},
            "test": {"it": "arc_it_test"},
        }

    def _create_problem(self, use_lm_eval=False, **kwargs):
        lm_tasks = self.lm_eval_tasks if use_lm_eval else None
        problem = CrossLingualMathProblem(
            merger=self.mock_merger,
            search_df=self.search_df,
            test_df=self.test_df,
            lang_id=self.lang_id,
            conf_pe=self.conf_pe,
            lm_eval_tasks=lm_tasks,
            n_var=3,
            n_obj=1,  # For MergingProblem, n_obj is 1
            **kwargs,
        )
        # Mock methods from BaseMergingProblem that are not part of this unit's focus
        problem.load_model = MagicMock(
            return_value=(
                ("mock_model", "mock_tokenizer")
                if not use_lm_eval
                else "mock_vllm_model"
            )
        )
        problem._from_array_to_genotype = MagicMock(
            return_value=Path("/fake/merged_model")
        )
        return problem

    def test_init_no_lm_eval(self):
        problem = self._create_problem(use_lm_eval=False)
        self.assertFalse(problem.use_lm_eval)
        self.assertEqual(problem.n_obj, 1)
        self.assertEqual(problem.lang_id, self.lang_id)

    def test_init_with_lm_eval(self):
        problem = self._create_problem(use_lm_eval=True)
        self.assertTrue(problem.use_lm_eval)
        self.assertEqual(problem.lm_eval_tasks, self.lm_eval_tasks)

    @patch("mergenetic.optimization.predefined_problems.get_batched_model_predictions")
    @patch("mergenetic.optimization.predefined_problems.FGMathEvaluator")
    @patch("mergenetic.optimization.predefined_problems.PerformanceEstimator")
    def test_metrics_4_genotype_no_lm_eval(
        self, MockPerformanceEstimator, MockFGMathEvaluator, mock_get_preds
    ):
        problem = self._create_problem(use_lm_eval=False)
        mock_model, mock_tokenizer = "fake_model", "fake_tokenizer"

        mock_get_preds.return_value = pd.Series(
            ["prompt_text_ans1"]
        )  # predictions after removing prompt
        mock_fg_eval_instance = MockFGMathEvaluator.return_value
        correctness_series_expected = pd.Series([True])
        mock_fg_eval_instance.get_correctness.return_value = correctness_series_expected

        mock_perf_est_instance = MockPerformanceEstimator.return_value
        mock_perf_est_instance.estimate_accuracy.return_value = 0.8

        fitness, desc = problem.metrics_4_genotype(mock_model, mock_tokenizer)

        mock_get_preds.assert_called_once()
        MockFGMathEvaluator.assert_called_once_with(language_id=self.lang_id)
        mock_fg_eval_instance.get_correctness.assert_called_once()
        MockPerformanceEstimator.assert_called_once()

        # Verify call to estimate_accuracy
        mock_perf_est_instance.estimate_accuracy.assert_called_once()
        called_args, _ = mock_perf_est_instance.estimate_accuracy.call_args
        pd.testing.assert_series_equal(
            called_args[0], correctness_series_expected, check_dtype=False
        )

        self.assertEqual(fitness, [-0.8])
        self.assertIn("Fitness value: 0.8", desc)

    @patch("mergenetic.optimization.predefined_problems.LmHarnessEvaluator")
    @patch("mergenetic.optimization.predefined_problems.PerformanceEstimator")
    def test_metrics_4_genotype_with_lm_eval(
        self, MockPerformanceEstimator, MockLmHarnessEvaluator
    ):
        problem = self._create_problem(use_lm_eval=True)
        mock_vllm_model = "fake_vllm_model"

        mock_lm_harness_instance = MockLmHarnessEvaluator.return_value
        correctness_series_expected = pd.Series([False, True])  # Correctness series
        mock_lm_harness_instance.evaluate.return_value = correctness_series_expected

        mock_perf_est_instance = MockPerformanceEstimator.return_value
        mock_perf_est_instance.estimate_accuracy.return_value = 0.6

        fitness, desc = problem.metrics_4_genotype(mock_vllm_model)

        MockLmHarnessEvaluator.assert_called_once_with(
            task_name=self.lm_eval_tasks["search"][self.lang_id],
            sample_ids=self.conf_pe.sample_ids,
            correctness_metric=self.conf_pe.correct_metric,
            lang_id=self.lang_id,  # detect_lang is True by default
            is_test=False,
            additional_templates_folder=None,
            batch_size=problem.eval_batch_size,
        )
        mock_lm_harness_instance.evaluate.assert_called_once_with(mock_vllm_model)
        MockPerformanceEstimator.assert_called_once()

        # Verify call to estimate_accuracy
        mock_perf_est_instance.estimate_accuracy.assert_called_once()
        called_args, _ = mock_perf_est_instance.estimate_accuracy.call_args
        pd.testing.assert_series_equal(
            called_args[0], correctness_series_expected, check_dtype=False
        )

        self.assertEqual(fitness, [-0.6])
        self.assertIn("Fitness value: 0.6", desc)

    def test_test_method_with_base_model(self):
        problem = self._create_problem(use_lm_eval=True)
        problem.metrics_4_genotype = MagicMock(return_value=([-0.9], "Test desc"))

        metrics, _ = problem.test(genotype=None, base_model=Path("/fake/base"))

        self.assertTrue(problem.test_mode)
        problem.load_model.assert_called_once_with(Path("/fake/base"))
        problem.metrics_4_genotype.assert_called_once_with("mock_vllm_model")
        self.assertEqual(metrics, [-0.9])

    def test_test_method_with_genotype(self):
        problem = self._create_problem(use_lm_eval=False)
        problem.metrics_4_genotype = MagicMock(return_value=([-0.75], "Test desc gen"))
        genotype = np.array([0.1, 0.2, 0.3])

        metrics, _ = problem.test(genotype=genotype)

        self.assertTrue(problem.test_mode)
        problem._from_array_to_genotype.assert_called_once_with(genotype)
        problem.load_model.assert_called_once_with(Path("/fake/merged_model"))
        problem.metrics_4_genotype.assert_called_once_with(
            "mock_model", "mock_tokenizer"
        )
        self.assertEqual(metrics, [-0.75])


class TestMultilingualMergingProblem(unittest.TestCase):
    def setUp(self):
        self.mock_merger = MagicMock()
        self.search_dfs = {
            "en": pd.DataFrame({"prompt": ["q_en"]}),
            "fr": pd.DataFrame({"prompt": ["q_fr"]}),
        }
        self.test_dfs = {
            "en": pd.DataFrame({"prompt": ["t_en"]}),
            "fr": pd.DataFrame({"prompt": ["t_fr"]}),
        }
        self.conf_pe = ConfigMultiLingualPE(
            sample_ids={"en": [0], "fr": [0]},
            weights={"en": [1.0], "fr": [1.0]},
            thetas=[[0.1]],
            bench="test_bench",
            mode="mean",
            correct_metric="acc",
        )
        self.lm_eval_tasks = {
            "search": {"en": "task_en_search", "fr": "task_fr_search"},
            "test": {"en": "task_en_test", "fr": "task_fr_test"},
        }

    def _create_problem(self, use_lm_eval=False, **kwargs):
        lm_tasks = self.lm_eval_tasks if use_lm_eval else None
        problem = MultilingualMergingProblem(
            merger=self.mock_merger,
            search_df_dict=self.search_dfs,
            test_df_dict=self.test_dfs,
            config_pe=self.conf_pe,
            lm_eval_tasks=lm_tasks,
            n_var=3,
            n_obj=2,  # n_obj must match number of languages/objectives
            **kwargs,
        )
        problem.load_model = MagicMock(
            return_value=(
                ("mock_model", "mock_tokenizer")
                if not use_lm_eval
                else "mock_vllm_model"
            )
        )
        problem._from_array_to_genotype = MagicMock(
            return_value=Path("/fake/merged_model")
        )
        return problem

    @patch("mergenetic.optimization.predefined_problems.get_batched_model_predictions")
    @patch("mergenetic.optimization.predefined_problems.MultilingualMCEvaluator")
    @patch("mergenetic.optimization.predefined_problems.PerformanceEstimator")
    def test_metrics_4_genotype_no_lm_eval(
        self, MockPerfEst, MockMultiMCEval, mock_get_preds
    ):
        problem = self._create_problem(use_lm_eval=False)
        mock_model, mock_tokenizer = "m", "t"

        mock_get_preds.side_effect = [pd.Series(["ans_en"]), pd.Series(["ans_fr"])]
        mock_multi_eval_inst = MockMultiMCEval.return_value
        mock_multi_eval_inst.get_correctness.return_value = {
            "en": pd.Series([True]),
            "fr": pd.Series([False]),
        }

        mock_perf_inst = MockPerfEst.return_value
        mock_perf_inst.estimate_accuracy.side_effect = [0.9, 0.4]  # en, fr

        fitness, desc = problem.metrics_4_genotype(mock_model, mock_tokenizer)

        self.assertEqual(mock_get_preds.call_count, 2)
        MockMultiMCEval.assert_called_once_with(
            language_ids=["en", "fr"], validate_lang=True
        )
        self.assertEqual(MockPerfEst.call_count, 2)
        self.assertEqual(fitness, [-0.9, -0.4])
        self.assertIn("'en': 0.9", desc)
        self.assertIn("'fr': 0.4", desc)

    @patch("mergenetic.optimization.predefined_problems.LmHarnessEvaluator")
    @patch("mergenetic.optimization.predefined_problems.PerformanceEstimator")
    def test_metrics_4_genotype_with_lm_eval(self, MockPerfEst, MockLmHarness):
        problem = self._create_problem(use_lm_eval=True)
        mock_vllm_model = "vllm"

        mock_lm_h_inst = MockLmHarness.return_value
        mock_lm_h_inst.evaluate.side_effect = [
            pd.Series([True]),
            pd.Series([False, True]),
        ]  # en, fr correctness

        mock_perf_inst = MockPerfEst.return_value
        mock_perf_inst.estimate_accuracy.side_effect = [0.85, 0.55]  # en, fr accuracy

        fitness, desc = problem.metrics_4_genotype(mock_vllm_model)

        self.assertEqual(MockLmHarness.call_count, 2)
        self.assertEqual(MockPerfEst.call_count, 2)
        self.assertEqual(fitness, [-0.85, -0.55])
        self.assertIn("'en': 0.85", desc)
        self.assertIn("'fr': 0.55", desc)


class TestLmEvalMultiObjectiveProblem(unittest.TestCase):
    def setUp(self):
        self.mock_merger = MagicMock()
        self.config = ConfigLmEvalMultiObjectivePE(
            tasks=["task_a", "task_b"],
            correct_metric="acc",
            sample_ids={"task_a": [0, 1], "task_b": [2, 3]},
        )

    def _create_problem(self, **kwargs):
        problem = LmEvalMultiObjectiveProblem(
            config=self.config,
            merger=self.mock_merger,
            n_var=3,
            n_obj=2,  # n_obj must match number of tasks
            **kwargs,
        )
        problem.load_model = MagicMock(return_value="mock_vllm_model")
        problem._from_array_to_genotype = MagicMock(
            return_value=Path("/fake/merged_model")
        )
        return problem

    def test_init(self):
        problem = self._create_problem()
        self.assertTrue(problem.use_lm_eval)
        self.assertEqual(problem.n_obj, 2)  # Based on len(config.tasks)

    @patch("mergenetic.optimization.predefined_problems.LmHarnessEvaluator")
    @patch("mergenetic.optimization.predefined_problems.PerformanceEstimator")
    def test_metrics_4_genotype(self, MockPerfEst, MockLmHarness):
        problem = self._create_problem()
        mock_vllm_model = "vllm"

        mock_lm_h_inst = MockLmHarness.return_value
        mock_lm_h_inst.evaluate.side_effect = [
            pd.Series([True]),
            pd.Series([False, True]),
        ]  # task_a, task_b

        mock_perf_inst = MockPerfEst.return_value
        mock_perf_inst.estimate_accuracy.side_effect = [0.7, 0.6]  # task_a, task_b

        fitness, desc = problem.metrics_4_genotype(mock_vllm_model)

        self.assertEqual(MockLmHarness.call_count, 2)
        # Check calls to LmHarnessEvaluator
        MockLmHarness.assert_any_call(
            task_name="task_a",
            sample_ids=self.config.sample_ids["task_a"],
            correctness_metric=self.config.correct_metric,
            is_test=False,
            additional_templates_folder=None,
            batch_size=problem.eval_batch_size,
        )
        MockLmHarness.assert_any_call(
            task_name="task_b",
            sample_ids=self.config.sample_ids["task_b"],
            correctness_metric=self.config.correct_metric,
            is_test=False,
            additional_templates_folder=None,
            batch_size=problem.eval_batch_size,
        )

        self.assertEqual(MockPerfEst.call_count, 2)
        # Check calls to PerformanceEstimator
        est_params_calls = [
            call_args[0][0] for call_args in MockPerfEst.call_args_list
        ]  # Get the est_params argument
        self.assertEqual(
            est_params_calls[0].sample_ids, self.config.sample_ids["task_a"]
        )
        self.assertEqual(
            est_params_calls[1].sample_ids, self.config.sample_ids["task_b"]
        )

        self.assertEqual(fitness, [-0.7, -0.6])
        self.assertIn("'task_a': 0.7", desc)
        self.assertIn("'task_b': 0.6", desc)

    def test_test_method_with_genotype(self):
        problem = self._create_problem()
        problem.metrics_4_genotype = MagicMock(
            return_value=([-0.9, -0.85], "Test desc")
        )
        genotype = np.array([0.5, 0.5, 0.5])

        metrics, _ = problem.test(genotype=genotype)

        self.assertTrue(problem.test_mode)
        problem._from_array_to_genotype.assert_called_once_with(genotype)
        problem.load_model.assert_called_once_with(
            Path("/fake/merged_model")
        )  # Path from _from_array_to_genotype mock
        problem.metrics_4_genotype.assert_called_once_with("mock_vllm_model")
        self.assertEqual(metrics, [-0.9, -0.85])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
