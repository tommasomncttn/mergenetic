from logging import getLogger

import numpy as np
import pandas as pd
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks import TaskManager

from .evaluator import LanguageDetector

logger = getLogger(__name__)


class LmHarnessEvaluator:
    """
    Evaluates the performance of a language model on a given task.

    Parameters
    ----------
    task_name : str
        Name of the task to evaluate.
    """

    task: ConfigurableTask
    task_manager: TaskManager = None
    sample_ids: list[int] | None
    lang_detector: LanguageDetector | None
    lang_id: str | None

    def __init__(
        self,
        task_name: str,
        correctness_metric: str = "exact_match",
        sample_ids: list[int] | None = None,
        lang_id: str | None = None,
        is_test: bool = False,
        additional_templates_folder: str | None = None,
        batch_size: int = 32,
    ) -> None:

        self.task_manager = TaskManager(include_path=additional_templates_folder)
        self.task: ConfigurableTask = self.get_task(task_name)

        self.sample_ids = sample_ids

        if sample_ids is not None and len(sample_ids) > 0:
            if is_test:
                # take all excluding sample_ids
                ids = np.arange(len(self.task.dataset["test"]))
                self.task.dataset["test"] = self.task.dataset["test"].select(
                    np.setdiff1d(ids, sample_ids)
                )

            else:
                self.task.dataset["test"] = self.task.dataset["test"].select(sample_ids)
                logger.debug("Sample ids provided. Using the specified sample ids.")
                logger.debug(f"Selected samples: {self.task.dataset['test']}")
        else:
            logger.info("No sample ids provided. Using the entire dataset.")

        if self.task.OUTPUT_TYPE == "multiple_choice":
            # disable language detection for multiple choice tasks
            logger.warning("Disabling language detection for multiple choice tasks.")
            lang_id = None

        try:
            self.lang_detector = (
                LanguageDetector([lang_id]) if lang_id is not None else None
            )
        except Exception as e:
            logger.warning(f"Language detection is disabled. Error: {e}")
            self.lang_detector = None

        self.correctness_metric = correctness_metric
        self.task_nm = task_name
        self.lang_id = lang_id
        self.batch_size = batch_size

    def get_task(self, task_name: str) -> ConfigurableTask:
        """
        Returns a task for the specified task name.

        Parameters
        ----------
        task_name : str
            Name of the task.

        Returns
        -------
        ConfigurableTask
            Task for the specified task.
        """
        return self.task_manager.load_task_or_group(task_name)[task_name]

    def evaluate(self, model: VLLM) -> list[float]:
        """
        Evaluates the performance of the model on the specified task.

        Parameters
        ----------
        model : VLLM
            The model to evaluate.

        Returns
        -------
        dict
            Evaluation results.
        """
        results = simple_evaluate(model, tasks=[self.task], batch_size=self.batch_size)
        # map results ids to sample ids
        if self.sample_ids is not None:
            for sample in results["samples"][self.task_nm]:
                sample["doc_id"] = self.sample_ids[sample["doc_id"]]

        def get_responses(results):
            answers = []
            logger.info(f"Num of Answers: {len(results['samples'][self.task_nm])}")

            for sample in results["samples"][self.task_nm]:
                if isinstance(sample["resps"], list) and len(sample["resps"]) > 0:
                    if (
                        isinstance(sample["resps"][0], list)
                        and len(sample["resps"][0]) > 0
                    ):
                        # flatten to a single list
                        list_resp = [
                            item for sublist in sample["resps"] for item in sublist
                        ]
                        answers.append(
                            {
                                "id": sample["doc_id"],
                                "correctness": sample[self.correctness_metric],
                                "model_answers": list_resp,
                            }
                        )
                    else:
                        answers.append(
                            {
                                "id": sample["doc_id"],
                                "correctness": sample[self.correctness_metric],
                                "model_answers": sample["resps"][0],
                            }
                        )
                else:
                    answers.append(
                        {
                            "id": sample["doc_id"],
                            "correctness": sample[self.correctness_metric],
                            "model_answers": sample["resps"],
                        }
                    )
            return answers

        self.data = pd.DataFrame(get_responses(results))

        logger.debug(
            f"Extracted {len(self.data)} samples from the model answers. Ids: {self.data['id']}"
        )

        # let's filter the answers by sample_ids
        self.data = (
            self.data[self.data["id"].isin(self.sample_ids)]
            if self.sample_ids is not None
            else self.data
        )

        # if the number of answers is more than len(sample_ids), then we need to filter the answers
        # by randomly picking one with the same id

        if self.sample_ids is not None and len(self.data) > len(self.sample_ids):
            logger.info(
                f"Number of samples in the dataset ({len(self.data)}) is greater than the number of sample ids provided ({len(self.sample_ids)})."
            )
            self.data = self.data.groupby("id").sample(n=1).reset_index(drop=True)

        if (len(self.data) != len(self.sample_ids)) and self.sample_ids is not None:
            logger.warning(
                f"Number of samples in the dataset ({len(self.data)}) does not match the number of sample ids provided ({len(self.sample_ids)})."
            )

        logger.debug(
            f"Samples from model answers: {self.data['model_answers'].head(5)}"
        )

        if self.lang_detector is None:
            logger.info(
                f"Language detection is disabled. Fitness: {self.data['correctness'].mean()}"
            )
            return self.data["correctness"]
        else:
            if isinstance(self.data["model_answers"][0], str):
                self.data["language"] = self.data["model_answers"].apply(
                    lambda x: self.lang_detector._get_language(
                        x[0] if not isinstance(x, str) else x
                    )
                )
                self.data["is_language_correct"] = self.data["language"] == (
                    "__label__" + self.lang_id
                )
            else:
                self.data["language"] = self.data["model_answers"].apply(
                    lambda x: [
                        self.lang_detector._get_language(
                            y[0] if not isinstance(y, str) else y
                        )
                        for y in x
                    ]
                )
                self.data["is_language_correct"] = self.data["language"].apply(
                    lambda x: avg_lang_correctness(self.lang_id, x)
                )

            logger.info(
                f"Language detection is enabled. Fitness: {(self.data['is_language_correct'] * self.data['correctness']).mean()}"
            )
            return self.data["is_language_correct"] * self.data["correctness"]

    def get_data(self) -> pd.DataFrame:
        """
        Returns the evaluation data.

        Returns
        -------
        pd.DataFrame
            Evaluation data.
        """
        return self.data


def avg_lang_correctness(lang_id: str, langs: list[str]) -> float:
    """
    Returns the average correctness of the language detection.

    Parameters
    ----------
    langs : list[str]
        List of languages.

    Returns
    -------
    float
        Average correctness of the language detection. Returns np.nan if langs is empty.
    """
    if not langs:  # Check if the list is empty
        return (
            np.nan
        )  # Return NaN directly to avoid np.mean warning, aligns with test expectation
    return np.mean([lang == "UNK" or lang == f"__label__{lang_id}" for lang in langs])
