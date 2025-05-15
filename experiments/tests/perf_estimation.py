import numpy as np
import pandas as pd
import pytest

from mergenetic.evaluation import (
    AccuracyEstimationParameters,
    AnchoredMultipleChoiceMathLanguageLoss,
    MultipleChoiceMathLanguageLoss,
)


@pytest.fixture
def sample_dataframe():
    """Returns a simple DataFrame for testing."""
    return pd.DataFrame(
        {
            "predictions": ["3.14", "42", "7", "not_a_number", ""],
            "answer": ["3.14", "42", "8", "not_a_number", "42"],
        }
    )


@pytest.fixture
def language_dataframe():
    """Returns a DataFrame with varied language predictions."""
    return pd.DataFrame(
        {
            "predictions": [
                "Ceci est une phrase",
                "Este es un texto",
                "Questo è italiano",
            ],
            "answer": ["test", "test", "test"],
        }
    )


@pytest.fixture
def accuracy_params():
    """Returns sample accuracy estimation parameters."""
    return AccuracyEstimationParameters(
        thetas=[np.array([0.5]), np.array([0.8])],
        sample_weights=np.array([1, 1, 1]),
        sample_ids=np.array([0, 1, 2]),
        bench="gsm8k",
        mode="mpirt",
        tb_data=None,
    )


def test_initialization(sample_dataframe):
    """Ensure that the class initializes correctly with a valid DataFrame."""
    evaluator = MultipleChoiceMathLanguageLoss(sample_dataframe)
    assert isinstance(evaluator.df, pd.DataFrame)


def test_missing_columns():
    """Ensure an error is raised if required columns are missing."""
    df = pd.DataFrame({"incorrect_col": ["3.14", "42"]})
    with pytest.raises(
        ValueError, match="DataFrame must contain 'predictions' and 'answer' columns."
    ):
        MultipleChoiceMathLanguageLoss(df)


def test_extract_numbers():
    """Test various number extraction cases."""
    assert MultipleChoiceMathLanguageLoss._extract_numbers("The answer is 42.") == "42"
    assert (
        MultipleChoiceMathLanguageLoss._extract_numbers("Pi is approximately 3.1415")
        == "3.1415"
    )
    assert MultipleChoiceMathLanguageLoss._extract_numbers("No numbers here!") is None


def test_language_detection(language_dataframe):
    """Ensure that language detection is working correctly."""
    evaluator = MultipleChoiceMathLanguageLoss(
        language_dataframe, fasttext_language_id="__label__it"
    )
    detected_languages = language_dataframe["predictions"].apply(
        evaluator._get_language
    )

    assert isinstance(detected_languages, pd.Series)
    assert len(detected_languages) == 3


def test_get_metrics(sample_dataframe):
    """Test accuracy computation for various prediction cases."""
    evaluator = MultipleChoiceMathLanguageLoss(sample_dataframe)
    accuracy = evaluator.get_metrics()
    assert 0.0 <= accuracy <= 1.0  # Ensure valid range


def test_get_metrics_with_dataframe(sample_dataframe):
    """Test if get_metrics() returns a DataFrame when requested."""
    evaluator = MultipleChoiceMathLanguageLoss(sample_dataframe)
    acc, df = evaluator.get_metrics(return_dataframe=True)

    assert isinstance(acc, float)
    assert isinstance(df, pd.DataFrame)
    assert "is_answer_correct" in df.columns


def test_anchored_evaluation(sample_dataframe, accuracy_params):
    """Ensure the anchored evaluator computes accuracy correctly."""
    evaluator = AnchoredMultipleChoiceMathLanguageLoss(
        sample_dataframe, accuracy_params
    )
    accuracy = evaluator.get_metrics()

    assert 0.0 <= accuracy <= 1.0  # Ensure it returns a valid probability
