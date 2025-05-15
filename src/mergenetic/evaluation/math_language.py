import re
from logging import getLogger
from typing import List, Optional, Union

import pandas as pd

from .evaluator import BaseEvaluator, LanguageDetector

logger = getLogger(__name__)


def extract_numbers(
    sentence: str, only_last_number: bool = True
) -> Optional[Union[str, List[str]]]:
    """
    Extracts numbers from a sentence.

    Parameters
    ----------
    sentence : str
        Input sentence.
    only_last_number : bool
        If True, return only the last number found.

    Returns
    -------
    str | list | None
        Extracted number(s) or None if no numbers found.
    """
    # Use a robust regex to find all sequences of digits, possibly with a decimal point,
    # ensuring they are whole numbers/decimals (bounded by word boundaries).
    all_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)

    if not all_numbers:
        return None

    if only_last_number:
        return all_numbers[-1]
    else:
        return all_numbers


# ==================================
#  MULTIPLE CHOICE EVALUATOR
# ==================================


class MCEvaluator(BaseEvaluator):
    """
    Evaluates the capabilities of an LLM for Multiple Choice questions in a specified language.
    """

    lang_detector: Optional[LanguageDetector]

    def __init__(self, language_id: str | None = "it") -> None:
        """
        Parameters
        ----------
        language_id : str
            Target language ID (default: "it" for Italian).
        """

        self.language_id = language_id
        self.lang_detector = None

        if self.language_id:
            try:
                self.lang_detector = LanguageDetector([self.language_id])
            except Exception as e:
                logger.warning(f"Language detection disabled: {e}")
                self.lang_detector = None

    def get_correctness(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Computes the correctness of model answers for Multiple Choice questions.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the model predictions and the correct answers.

        Returns
        -------
        pd.Series
            Returns the correctness of the answers.
        """
        self._validate_dataframe(dataframe)
        self.data = dataframe.copy()

        if not self.lang_detector:  # Check if lang_detector is available
            self.data = self.data.assign(
                filtered_answer=self.data["predictions"].apply(extract_numbers),
            )
            self.data["correctness"] = self.data["filtered_answer"] == self.data[
                "answer"
            ].astype(str)
            return self.data["correctness"]
        else:
            # Use list comprehension for language detection to avoid potential apply issues with mocks
            langs = [
                self.lang_detector._get_language(p) for p in self.data["predictions"]
            ]
            self.data = self.data.assign(
                language=pd.Series(langs, index=self.data.index),
                filtered_answer=self.data["predictions"].apply(extract_numbers),
            )
            self.data = self.data.assign(
                is_language_correct=lambda df: (
                    df["language"] == f"__label__{self.language_id}"
                )
                | (df["language"] == "UNK"),
                is_answer_correct=lambda df: df["filtered_answer"]
                == df["answer"].astype(str),
            )
            self.data["correctness"] = (
                self.data["is_language_correct"] & self.data["is_answer_correct"]
            )
            return self.data["correctness"]

    def get_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame with the results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the evaluation.
        """
        return self.data


# ==================================
#  MATH FREE GENERATION EVALUATOR
# ==================================


class FGMathEvaluator(BaseEvaluator):
    """
    Evaluates the capabilities of an LLM in solving math word problems in a specified language.
    """

    lang_detector: Optional[LanguageDetector]  # Allow Optional for consistency

    def __init__(self, language_id: str | None = None) -> None:
        """
        Parameters
            ----------
        language_id : str
            Target language ID (default: "it" for Italian).
        """

        self.language_id = language_id
        self.lang_detector = None  # Initialize to None

        if self.language_id:  # Use self.language_id here
            try:
                self.lang_detector = LanguageDetector([self.language_id])
            except Exception as e:
                logger.warning(f"Language detection for FGMathEvaluator disabled: {e}")
                self.lang_detector = None

    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Extracts the last number from the model's prediction.

        Parameters
        ----------
        text : str
            The text from which to extract the answer.

        Returns
        -------
        str
            The extracted number, or 'None' if no match is found.
        """
        matches = re.findall(r"\d+", text)
        return matches[-1] if matches else "None"

    def get_correctness(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Computes the accuracy of the model for math word problems.

        Parameters
        ----------
        return_dataframe : bool, optional
            If True, return the full evaluation DataFrame.

        Returns
        -------
        dict[str, float] | tuple[dict[str, float], dict[str, pd.DataFrame]]
            Accuracy for each language, and optionally the evaluation DataFrame.
        """
        self._validate_dataframe(dataframe)

        self.data = dataframe.copy()
        self.data["predictions_filtered"] = self.data["predictions"].apply(
            self._extract_answer
        )
        self.data["correctness"] = self.data["predictions_filtered"] == self.data[
            "answer"
        ].astype(str)

        if self.lang_detector:  # Check if lang_detector is available
            # Use list comprehension for language detection
            langs = [
                self.lang_detector._get_language(p) for p in self.data["predictions"]
            ]
            self.data["language"] = pd.Series(langs, index=self.data.index)

            self.data["evaluation_log"] = self.data.apply(
                lambda row: f"Predicted: {row.predictions_filtered}, Actual: {row.answer}, Correct: {row.correctness}\nFull Answer: {row.predictions}, Language: {row.language}",
                axis=1,
            )

            # Combine correctness with language correctness
            lang_correct = (self.data["language"] == f"__label__{self.language_id}") | (
                self.data["language"] == "UNK"
            )
            return self.data["correctness"] & lang_correct
        else:
            self.data["evaluation_log"] = self.data.apply(
                lambda row: f"Predicted: {row.predictions_filtered}, Actual: {row.answer}, Correct: {row.correctness}\nFull Answer: {row.predictions}",
                axis=1,
            )
            return self.data["correctness"]

    def get_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame with the results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted numbers.
        """
        return self.data
