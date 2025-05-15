import re
from logging import getLogger
from typing import Dict, List, Tuple, Union

import pandas as pd

from .evaluator import BaseMultiObjectiveEvaluator, LanguageDetector

logger = getLogger(__name__)

# =========================================
# MULTILINGUAL MULTIPLE CHOICE EVALUATOR
# =========================================


class MultilingualMCEvaluator(BaseMultiObjectiveEvaluator):
    """
    Base class for multiple-choice question evaluators, providing common methods for
    extracting answers for multiple-choice questions, detecting languages, and computing accuracy.
    """

    lang_detector: LanguageDetector

    def __init__(self, language_ids: list[str] = None, validate_lang=True) -> None:
        """
        Initializes the evaluator with a dictionary of dataframes, each corresponding to a language.

        Parameters
        ---------
        language_ids : list[str]
            List of language IDs for Language Detection.
        validate_lang : bool
            If True, validates the language of the predictions.
        """

        self.language_ids = language_ids
        self.validate_lang = validate_lang

        if validate_lang:
            try:
                self.lang_detector = LanguageDetector(language_ids)
            except Exception as e:
                logger.warning(f"Language detection disabled: {e}")
                self.lang_detector = None
        else:
            self.lang_detector = None

    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Extracts the selected option from the model's prediction.

        Parameters
        ----------
        text : str
            The text from which to extract the answer.

        Returns
        -------
        str
            The extracted answer option (e.g., 'A', 'B', 'C', 'D'), or 'None' if no match is found.
        """
        matches = re.findall(r"\(?([A-Z])\)", text)
        return matches[-1] if matches else "None"

    def get_correctness(
        self, dataframe_dict: Dict[str, pd.DataFrame]
    ) -> Union[float, Tuple[List[float], str]]:
        """
        Computes correctness across multiple language datasets.

        Parameters
        ----------
        dataframe_dict : dict[str, pd.DataFrame]
            Dictionary of dataframes, each corresponding to a language.

        Returns
        -------
        float | tuple[list[float], str]
            Per-language correctness.
        """
        self.data: dict[str, pd.DataFrame] = {}
        for k, v in dataframe_dict.items():
            self._validate_dataframe(v)
            self.data[k] = v.copy()

        for lang, df in self.data.items():
            df["predictions_filtered"] = df["predictions"].apply(self._extract_answer)
            df["language"] = df["predictions"].apply(self.lang_detector._get_language)

        def calc_correctness(df: pd.DataFrame, lang: str = None):
            if self.validate_lang:
                if self.lang_detector is None:
                    raise ValueError(
                        "Language detector must be provided for language validation."
                    )

                correctness = (df["predictions_filtered"] == df["answer"]) & (
                    (df["language"] == f"__label__{lang}") | (df["language"] == "UNK")
                )

            else:
                correctness = df["predictions_filtered"] == df["answer"]

            df["is_correct"] = correctness
            return correctness

        correctness_dict: Dict[str, pd.Series] = {}
        for lang, df in self.data.items():
            correctness_dict[lang] = calc_correctness(df, lang)

        return correctness_dict

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the DataFrame with the results.

        Returns
        -------
        dict[str, pd.DataFrame]
            The DataFrame with the results.
        """
        return self.data


# ==============================================
# MULTILINGUAL MATH FREE GENERATION EVALUATOR
# ==============================================
class MultilingualMathFGEvaluator(BaseMultiObjectiveEvaluator):
    """
    Evaluator for multilingual math datasets on model generated answers.
    """

    lang_detector: LanguageDetector

    def __init__(
        self, language_ids: list[str] = [], validate_lang: bool = True
    ) -> None:
        """
        Parameters
        ----------
        fasttext_model_path : str
            Path to the FastText model for language detection.

        """
        self.language_ids = language_ids
        self.validate_lang = validate_lang

        if validate_lang:
            self.lang_detector = LanguageDetector(language_ids)

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
        dataframes_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
        Computes the correctness of the model for math problems with free generation.

        Parameters
        ----------
        dataframes_dict : dict[str, pd.DataFrame]
            Dictionary of dataframes, each corresponding to a language.

        Returns
        -------
        dict[str, float] | tuple[dict[str, float], dict[str, pd.DataFrame]]
            Per-language correctness.
        """
        correctness_dict = {}

        self.data: Dict[str, pd.DataFrame] = {}
        for k, v in dataframes_dict.items():
            self._validate_dataframe(v)
            self.data[k] = v.copy()

        for df in self.data.values():
            df["predictions_filtered"] = df["predictions"].apply(self._extract_answer)
            df["is_correct"] = df["predictions_filtered"] == df["answer"]

            if self.validate_lang:
                df["language"] = df["predictions"].apply(
                    self.lang_detector._get_language
                )

        def calc_correctness(df: pd.DataFrame, lang: str = None):
            if not self.validate_lang:
                return df["is_correct"]
            else:
                return df["is_correct"] & (
                    (df["language"] == f"__label__{lang}") | (df["language"] == "UNK")
                )

        for lang, df in self.data.items():
            correctness_dict[lang] = calc_correctness(df, lang)

        return correctness_dict

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the DataFrame with the results.

        Returns
        -------
        dict[str, pd.DataFrame]
            The DataFrame with the results.
        """
        return self.data
