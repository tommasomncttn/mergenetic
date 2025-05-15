from abc import ABC, abstractmethod
from typing import Dict

import fasttext
import pandas as pd

from mergenetic import PROJECT_ROOT

# ==========================
#  BASE EVALUATOR CLASSES
# ==========================


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """

    data: pd.DataFrame

    @abstractmethod
    def get_correctness(self, dataframe: pd.DataFrame, **kwargs):
        """
        Abstract method to return computed correctness.
        """
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Abstract method to return the predictions DataFrame.
        """
        pass

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Ensures that the DataFrame has the required columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if "predictions" not in df.columns or "answer" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'predictions' and 'answer' columns."
            )


class BaseMultiObjectiveEvaluator(BaseEvaluator):
    """
    Abstract base class for multi-objective evaluators.
    """

    data: Dict[str, pd.DataFrame]

    @abstractmethod
    def get_correctness(self, dataframes_dict: Dict[str, pd.DataFrame], **kwargs):
        """
        Abstract method to return computed correctness.
        """
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, pd.DataFrame]:
        """
        Abstract method to return a dictionary of predictions DataFrames.
        """
        pass


# =====================
#  LANGUAGE DETECTOR
# =====================
class LanguageDetector:
    """
    Detects the language of a given text using FastText.
    """

    SUPPORTED_LANGUAGES = set(
        """
        af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv 
        cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu 
        hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr 
        min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt 
        qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz 
        vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
    """.split()
    )

    def __init__(
        self,
        langs: list[str],
        fasttext_model_path: str = f"{PROJECT_ROOT}/models/fasttext/lid.176.bin",
    ):
        """
        Initializer for the language detector.

        Parameters
        ----------
        langs: list[str]
            List with languages used for language detection.
        fasttext_model_path: str
            Path to the FastText model.
        """
        self.detector = fasttext.load_model(fasttext_model_path)

        unsupported_languages = [
            lang for lang in langs if lang not in self.SUPPORTED_LANGUAGES
        ]
        if unsupported_languages:
            raise ValueError(
                f"Unsupported languages detected: {unsupported_languages}."
            )

    def _get_language(self, text: str) -> str:
        """
        Detects the language of a given text using FastText.

        Parameters
        ----------
        text : str
            The text to detect the language from.

        Returns
        -------
        str
            Detected language label or "UNK" if confidence is too low.
        """
        text_clean = text.replace("\n", " ")
        pred_lang, confidence = self.detector.predict(text_clean)

        if confidence[0] < 0.3:
            return "UNK"
        elif confidence[0] < 0.5:
            return self.detector.predict(" ".join(text_clean.split()[1:]))[0]
        return pred_lang[0] if confidence[0] >= 0.5 else "UNK"

    def get_supported_languages(self) -> set[str]:
        """
        Returns the set of supported languages.

        Returns
        -------
        set[str]
            Set of supported languages.
        """
        return self.SUPPORTED_LANGUAGES
