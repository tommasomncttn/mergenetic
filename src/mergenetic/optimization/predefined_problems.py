import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from mergenetic.estimator.perf_estimation import (
    PerformanceEstimationParameters,
    PerformanceEstimator,
)
from mergenetic.evaluation.lm_harness import LmHarnessEvaluator
from mergenetic.evaluation.math_language import FGMathEvaluator
from mergenetic.evaluation.multilingual_evaluator import MultilingualMCEvaluator
from mergenetic.optimization import MergingProblem, MultiObjectiveMergingProblem
from mergenetic.utils import get_batched_model_predictions

logger = logging.getLogger(__name__)


@dataclass
class ConfigPE:
    thetas: list[list[float]]
    sample_ids: list[int]
    weights: list[float]
    correct_metric: str = None
    bench: str = field(default=None)
    mode: str = field(default=None)


class CrossLingualMathProblem(MergingProblem):
    """
    Class for optimizating merged models in order to transfer skills across different languages.
    """

    def __init__(
        self,
        merger,
        search_df: pd.DataFrame | None,
        test_df: pd.DataFrame | None,
        lang_id: str,
        conf_pe: ConfigPE,
        lm_eval_tasks: dict[str, str] | None = None,
        n_var: int = 11,
        n_obj: int = 2,
        n_eq_constr: int = 0,
        n_ieq_constr: int = 0,
        xl=0,
        xu=1,
        eval_batch_size: int = 5,
        detect_lang=True,
        additional_templates_folder: str | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        merger : Merger
            The merger to use for merging models.
        search_df : pd.DataFrame
            The dataframe to use for searching for the best model.
        test_df : pd.DataFrame
            The dataframe to use for testing the best model.
        lm_eval_tasks : dict[str, str]
            Dictionary of tasks to evaluate the model on, each corresponding to a search or test phase.
        lang_id : str
            Identifier for the language being processed.
        n_var : int
            Number of variables.
        n_obj : int
            Number of objectives.
        n_eq_constr : int
            Number of equality constraints.
        n_ieq_constr : int
            Number of inequality constraints.
        xl : int
            Lower bound for the variables.
        xu : int
            Upper bound for the variables.
        eval_batch_size : int
            Batch size for evaluation.
        """

        # init the parent with the right params
        super().__init__(
            merger=merger,
            search_df=search_df,
            test_df=test_df,
            n_var=n_var,
            n_obj=n_obj,
            n_eq_constr=n_eq_constr,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            eval_batch_size=eval_batch_size,
            use_lm_eval=lm_eval_tasks is not None,
            **kwargs,
        )

        self.test_df = test_df
        self.lm_eval_tasks = lm_eval_tasks
        self.lang_id = lang_id
        self.conf_pe = conf_pe
        self.detect_lang = detect_lang
        self.additional_templates_folder = additional_templates_folder

    def metrics_4_genotype(
        self, model, tokenizer: Optional[object] = None
    ) -> Union[list[float], str]:
        """
        Method to evaluate the performance of the merged model/genotype.
        A prediction is correct if in italian and the last number in the string is the correct answer.

        Parameters
        ----------
        model : Model
            The model to evaluate.
        tokenizer : Tokenizer
            The tokenizer to use for the model. Not required if self.use_lm_eval is True.
        Returns
        -------
        list[float]
            A list of metrics that will be used to evaluate the performance of the merged model in out["F"]
        """

        # get the predictions
        device = self.device if self.device is not None else "cpu"

        df = self.test_df if self.test_mode else self.search_df

        if not self.use_lm_eval:
            df["predictions"] = get_batched_model_predictions(
                model,
                tokenizer,
                df,
                batch_size=self.eval_batch_size,
                max_token=256,
                device=device,
            )
            # remove prompt text in column 'prompt' from 'predictions'
            df["predictions"] = df.apply(
                lambda row: row["predictions"].replace(row["prompt"], ""), axis=1
            )

            # get the loss with the evaluator
            evaluator = FGMathEvaluator(language_id=self.lang_id)
            correctness = evaluator.get_correctness(df)
        else:
            evaluator = LmHarnessEvaluator(
                task_name=self.lm_eval_tasks[
                    "search" if not self.test_mode else "test"
                ][self.lang_id],
                sample_ids=self.conf_pe.sample_ids,
                correctness_metric=self.conf_pe.correct_metric,
                lang_id=self.lang_id if self.detect_lang else None,
                is_test=self.test_mode,
                additional_templates_folder=self.additional_templates_folder,
                batch_size=self.eval_batch_size,
            )
            correctness = evaluator.evaluate(model)

        est_params = PerformanceEstimationParameters(
            thetas=self.conf_pe.thetas,
            sample_weights=self.conf_pe.weights,
            sample_ids=self.conf_pe.sample_ids,
            mode=self.conf_pe.mode if not self.test_mode else "mean",
            bench=self.conf_pe.bench,
        )

        f = -(PerformanceEstimator(est_params).estimate_accuracy(correctness))

        return [f], f"Fitness value: {-f}"

    def test(
        self, genotype, base_model: Optional[Path] = None
    ) -> Union[list[float], str]:
        """
        Evaluate a model's performance on a test dataset.

        Parameters
        ----------
        genotype : list
            The genotype to evaluate.
        base_model : Path, optional
            The path to the base model to use for evaluation.

        Returns
        -------
        Union[list[float], str]
            The metrics for the model.
        """
        self.test_mode = True

        if base_model:
            if self.use_lm_eval:
                model = self.load_model(base_model)
            else:
                model, tokenizer = self.load_model(base_model)
        else:
            assert (
                len(genotype) == self.n_var
            ), f"Genotype length mismatch: expected {self.n_var}, got {len(genotype)}."
            path_to_model = self._from_array_to_genotype(genotype)

            if self.use_lm_eval:
                model = self.load_model(path_to_model)
            else:
                model, tokenizer = self.load_model(path_to_model)

        if self.use_lm_eval:
            return self.metrics_4_genotype(model)
        else:
            return self.metrics_4_genotype(model, tokenizer)


# =====================
#  MULTILINGUAL PROBLEM
# =====================
@dataclass
class ConfigMultiLingualPE:
    sample_ids: dict[str, list[int]]
    weights: dict[str, list[float]]
    thetas: list[list[float]]
    bench: str
    mode: str
    correct_metric: str = None


class MultilingualMergingProblem(MultiObjectiveMergingProblem):
    """
    Class for optimizating merged models on multiple languages.
    """

    def __init__(
        self,
        merger,
        search_df_dict: dict[str, pd.DataFrame] | None,
        test_df_dict: dict[str, pd.DataFrame] | None,
        config_pe: ConfigMultiLingualPE,
        lm_eval_tasks: dict[str, dict[str]] | None = None,
        n_var: int = 11,
        n_obj: int = 2,
        n_eq_constr: int = 0,
        n_ieq_constr: int = 0,
        xl=0,
        xu=1,
        eval_batch_size: int = 5,
        detect_lang=True,
        additional_templates_folder: str | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        merger : Merger
            The merger to use for merging models.
        dataframe_dictionary : dict[str, pd.DataFrame]
            Dictionary of dataframes, each corresponding to a language.
        test_df_dict : dict[str, pd.DataFrame]
            Dictionary of test dataframes, each corresponding to a language.
        lm_eval_tasks : dict[str, dict[str]] | None
            Dictionary of tasks to evaluate the model on, each corresponding to search and test.
        config_pe : ConfigMultiLingualPE
            Configuration for performance estimation.
        n_var : int
            Number of variables.
        n_obj : int
            Number of objectives.
        n_eq_constr : int
            Number of equality constraints.
        n_ieq_constr : int
            Number of inequality constraints.
        xl : int
            Lower bound for the variables.
        xu : int
            Upper bound for the variables.

        Returns
        -------
        None
        """

        # init the parent with the right params
        super().__init__(
            merger=merger,
            search_dataframes=search_df_dict,
            test_dataframes=test_df_dict,
            n_var=n_var,
            n_obj=n_obj,
            n_eq_constr=n_eq_constr,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            eval_batch_size=eval_batch_size,
            use_lm_eval=lm_eval_tasks is not None,
            **kwargs,
        )

        self.search_df_dict = search_df_dict
        self.test_df_dict = test_df_dict
        self.lm_eval_tasks = lm_eval_tasks
        self.conf_pe = config_pe
        self.detect_lang = detect_lang
        self.additional_templates_folder = additional_templates_folder

    def metrics_4_genotype(self, model, tokenizer=None) -> Union[list[float], str]:
        """
        Method to evaluate the performance of the merged model/genotype.
        A prediction is correct if in italian and the last number in the string is the correct answer.

        Parameters
        ----------
        model : Model
            The model to evaluate.
        tokenizer : Tokenizer
            The tokenizer to use for the model.
        Returns
        -------
        Union[list[float], str]
            A list of metrics that will be used to evaluate the performance of the merged model in out["F"]
        """
        device = self.device if self.device is not None else "cpu"
        dfs = self.test_df_dict if self.test_mode else self.search_df_dict

        if not self.use_lm_eval:
            # get predictions
            for k, data in dfs.items():
                data["predictions"] = get_batched_model_predictions(
                    model,
                    tokenizer,
                    data,
                    batch_size=self.eval_batch_size,
                    device=device,
                    print_output=True,
                )
                # remove prompt text in column 'prompt' from 'predictions'
                data["predictions"] = data.apply(
                    lambda row: row["predictions"].replace(row["prompt"], ""), axis=1
                )

            evaluator = MultilingualMCEvaluator(
                language_ids=list(dfs.keys()), validate_lang=True
            )
            correctness_dict = evaluator.get_correctness(dfs)

        else:
            correctness_dict = {}
            tasks = self.lm_eval_tasks["search" if not self.test_mode else "test"]
            for lang in self.conf_pe.sample_ids.keys():
                evaluator = LmHarnessEvaluator(
                    task_name=tasks[lang],
                    sample_ids=self.conf_pe.sample_ids[lang],
                    correctness_metric=self.conf_pe.correct_metric,
                    lang_id=lang if self.detect_lang else None,
                    is_test=self.test_mode,
                    additional_templates_folder=self.additional_templates_folder,
                    batch_size=self.eval_batch_size,
                )
                correctness_dict[lang] = evaluator.evaluate(model)

        # get metrics
        acc_dict = {}
        for k, correctness in correctness_dict.items():
            est_params = PerformanceEstimationParameters(
                thetas=self.conf_pe.thetas,
                sample_weights=self.conf_pe.weights[k],
                sample_ids=self.conf_pe.sample_ids[k],
                mode=self.conf_pe.mode if not self.test_mode else "mean",
                bench=self.conf_pe.bench,
            )
            perf_estimator = PerformanceEstimator(est_params)

            logger.info(f"Correctness for {k}: {correctness}")
            acc_dict[k] = perf_estimator.estimate_accuracy(correctness)

        f = [-1 * acc for acc in acc_dict.values()]
        description = "Fitness values: " + str(acc_dict)

        if not f:
            raise ValueError("No metrics were computed.")

        return f, description

    def test(
        self, genotype, base_model: Optional[Path] = None
    ) -> Union[list[float], str]:
        """
        Evaluate a model's performance on a test dataset.

        Parameters
        ----------
        genotype : list
            The genotype to evaluate.
        base_model : Path, optional
            The path to the base model to use for evaluation.

        Returns
        -------
        Union[list[float], str]
            The metrics for the model.
        """
        self.test_mode = True

        if base_model:
            if self.use_lm_eval:
                model = self.load_model(base_model)
            else:
                model, tokenizer = self.load_model(base_model)
        else:
            assert (
                len(genotype) == self.n_var
            ), f"Genotype length mismatch: expected {self.n_var}, got {len(genotype)}."
            path_to_model = self._from_array_to_genotype(genotype)

            if self.use_lm_eval:
                model = self.load_model(path_to_model)
            else:
                model, tokenizer = self.load_model(path_to_model)

        if self.use_lm_eval:
            return self.metrics_4_genotype(model)
        else:
            return self.metrics_4_genotype(model, tokenizer)


# ===============================
#  LM-EVAL MULTIOBJECTIVE PROBLEM
# ===============================
@dataclass
class ConfigLmEvalMultiObjectivePE:
    tasks: list[str]
    correct_metric: str
    sample_ids: dict[str, list[int]]
    additional_templates_folder: str | None = None

    def __post_init__(self):
        if not isinstance(self.tasks, list):
            raise ValueError("Tasks should be a list of task names.")
        if not isinstance(self.sample_ids, dict):
            raise ValueError("Sample IDs should be a dictionary of lists of integers.")
        if not isinstance(self.correct_metric, str):
            raise ValueError("Correctness metric should be a string.")
        if not isinstance(self.additional_templates_folder, (str, type(None))):
            raise ValueError("Additional templates folder should be a string or None.")


class LmEvalMultiObjectiveProblem(MultiObjectiveMergingProblem):
    """
    Class for evolving merged models.
    """

    def __init__(
        self,
        config: ConfigLmEvalMultiObjectivePE,
        merger,
        n_var: int = 11,
        n_obj: int = 2,
        n_eq_constr: int = 0,
        n_ieq_constr: int = 0,
        xl=0,
        xu=1,
        **kwargs,
    ):
        super().__init__(
            merger=merger,
            n_var=n_var,
            n_obj=n_obj,
            n_eq_constr=n_eq_constr,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            search_dataframes=None,
            test_dataframes=None,
            use_lm_eval=True,
            **kwargs,
        )

        self.config = config

    def metrics_4_genotype(self, model) -> Union[list[float], str]:
        """
        Method to evaluate the performance of the merged model/genotype.

        Parameters
        ----------
        model : Model
            The model to evaluate.
        Returns
        -------
        Union[list[float], str]
            A list of metrics that will be used to evaluate the performance of the merged model in out["F"]
        """

        correctness_dict = {}
        for task in self.config.tasks:
            evaluator = LmHarnessEvaluator(
                task_name=task,
                sample_ids=self.config.sample_ids[task],
                correctness_metric=self.config.correct_metric,
                is_test=self.test_mode,
                additional_templates_folder=self.config.additional_templates_folder,
                batch_size=self.eval_batch_size,
            )
            correctness_dict[task] = evaluator.evaluate(model)

        # get metrics
        acc_dict = {}
        for k, correctness in correctness_dict.items():
            est_params = PerformanceEstimationParameters(
                thetas=None,
                sample_weights=np.ones(len(self.config.sample_ids[k]))
                / len(self.config.sample_ids[k]),
                sample_ids=self.config.sample_ids[k],
                mode="mean",
                bench=None,
            )
            perf_estimator = PerformanceEstimator(est_params)
            logger.info(f"Correctness for {k}: {correctness.to_list()}")
            acc_dict[k] = perf_estimator.estimate_accuracy(correctness)

        f = [-1 * acc for acc in acc_dict.values()]
        description = "Fitness values: " + str(acc_dict)

        if not f:
            raise ValueError("No metrics were computed.")

        return f, description

    def test(
        self, genotype, base_model: Optional[Path] = None
    ) -> Union[list[float], str]:
        """
        Evaluate a model's performance on a test dataset.
        Parameters
        ----------
        genotype : list
            The genotype to evaluate.
        base_model : Path, optional
            The path to the base model to use for evaluation.
        Returns
        -------
        Union[list[float], str]
            The metrics for the model.
        """
        self.test_mode = True
        if base_model:
            model = self.load_model(base_model)
        else:
            assert (
                len(genotype) == self.n_var
            ), f"Genotype length mismatch: expected {self.n_var}, got {len(genotype)}."
            path_to_model = self._from_array_to_genotype(genotype)

        model = self.load_model(path_to_model)

        return self.metrics_4_genotype(model)
