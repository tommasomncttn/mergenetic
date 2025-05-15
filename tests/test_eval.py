import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from mergenetic.evaluation.math_language import (
    FGMathEvaluator,
    MCEvaluator,
    extract_numbers,
)


class TestExtractNumbers(unittest.TestCase):
    def test_only_last_number_true_multiple_numbers(self):
        self.assertEqual(
            extract_numbers("Sentence with 123 and 45.67 then 890."), "890"
        )

    def test_only_last_number_true_single_number(self):
        self.assertEqual(extract_numbers("Only 1 number 123 here."), "123")

    def test_only_last_number_true_number_at_start(self):
        self.assertEqual(extract_numbers("789 is at the start."), "789")

    def test_only_last_number_true_number_at_end(self):
        self.assertEqual(extract_numbers("The number is 90.12"), "90.12")
        self.assertEqual(extract_numbers("The number is 90"), "90")

    def test_only_last_number_true_no_numbers(self):
        self.assertIsNone(extract_numbers("No numbers here."))

    def test_only_last_number_true_numbers_with_letters(self):
        self.assertEqual(
            extract_numbers("Number 123abc should be 456"), "456"
        )  # 123abc is not matched due to (?!\w)
        self.assertEqual(extract_numbers("Number 123.45abc should be 678.90"), "678.90")

    def test_only_last_number_false_multiple_numbers(self):
        self.assertEqual(
            extract_numbers(
                "Sentence with 123 and 45.67 then 890.", only_last_number=False
            ),
            ["123", "45.67", "890"],
        )

    def test_only_last_number_false_single_number(self):
        # Adjusted expectation: extract_numbers should find all numbers. "1" is a number.
        self.assertEqual(
            extract_numbers("Only 1 number 123 here.", only_last_number=False),
            ["1", "123"],
        )

    def test_only_last_number_false_no_numbers(self):
        self.assertIsNone(extract_numbers("No numbers here.", only_last_number=False))

    def test_only_last_number_false_float_and_int(self):
        self.assertEqual(
            extract_numbers("Value 1.23 and 456.", only_last_number=False),
            ["1.23", "456"],
        )


class TestMCEvaluator(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "predictions": [
                    "Answer is 7",
                    "The correct one is 3.14",
                    "No idea, maybe 0",
                ],
                "answer": ["7", "3.14", "1"],
            }
        )
        self.lang_id_it = "it"

    @patch("mergenetic.evaluation.math_language.LanguageDetector")
    def test_init_with_language_id(self, mock_lang_detector_class):
        mock_detector_instance = MagicMock()
        mock_lang_detector_class.return_value = mock_detector_instance

        evaluator = MCEvaluator(language_id=self.lang_id_it)

        self.assertEqual(evaluator.language_id, self.lang_id_it)
        mock_lang_detector_class.assert_called_once_with([self.lang_id_it])
        self.assertEqual(evaluator.lang_detector, mock_detector_instance)

    def test_init_with_language_id_none(self):
        evaluator = MCEvaluator(language_id=None)
        self.assertIsNone(evaluator.language_id)
        self.assertIsNone(evaluator.lang_detector)

    @patch("mergenetic.evaluation.math_language.logger")
    @patch(
        "mergenetic.evaluation.math_language.LanguageDetector",
        side_effect=ValueError("Test Detector Error"),
    )
    def test_init_language_detector_fails(self, mock_lang_detector_class, mock_logger):
        evaluator = MCEvaluator(language_id=self.lang_id_it)

        self.assertIsNone(evaluator.lang_detector)
        mock_logger.warning.assert_called_once_with(
            "Language detection disabled: Test Detector Error"
        )

    @patch.object(
        MCEvaluator, "_validate_dataframe"
    )  # Patching the method on the class
    def test_get_correctness_no_language_id(self, mock_validate_df):
        evaluator = MCEvaluator(language_id=None)
        correctness = evaluator.get_correctness(
            self.sample_df.copy()
        )  # Use copy to avoid modifying original

        mock_validate_df.assert_called_once()
        expected_filtered_answer = pd.Series(["7", "3.14", "0"], name="filtered_answer")
        expected_correctness = pd.Series([True, True, False], name="correctness")

        pd.testing.assert_series_equal(
            evaluator.data["filtered_answer"],
            expected_filtered_answer,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            correctness, expected_correctness, check_names=False
        )

    @patch.object(MCEvaluator, "_validate_dataframe")
    @patch("mergenetic.evaluation.math_language.LanguageDetector")
    def test_get_correctness_with_language_id(
        self, mock_lang_detector_class, mock_validate_df
    ):
        mock_detector_instance = MagicMock()
        mock_detector_instance._get_language.side_effect = [
            "__label__it",
            "__label__it",
            "UNK",
        ]  # Mocked lang detection
        mock_lang_detector_class.return_value = mock_detector_instance

        evaluator = MCEvaluator(language_id=self.lang_id_it)
        self.assertIsNotNone(evaluator.lang_detector)
        self.assertEqual(
            evaluator.lang_detector, mock_detector_instance
        )  # Verify mock is set

        correctness = evaluator.get_correctness(self.sample_df.copy())

        mock_validate_df.assert_called_once()
        self.assertEqual(
            mock_detector_instance._get_language.call_count, len(self.sample_df)
        )

        expected_language = pd.Series(
            ["__label__it", "__label__it", "UNK"], name="language"
        )
        expected_filtered_answer = pd.Series(["7", "3.14", "0"], name="filtered_answer")
        expected_is_lang_correct = pd.Series(
            [True, True, True], name="is_language_correct"
        )  # UNK is also correct
        expected_is_answer_correct = pd.Series(
            [True, True, False], name="is_answer_correct"
        )
        expected_final_correctness = pd.Series(
            [True, True, False], name="correctness"
        )  # True & True, True & True, True & False

        pd.testing.assert_series_equal(
            evaluator.data["language"], expected_language, check_names=False
        )
        pd.testing.assert_series_equal(
            evaluator.data["filtered_answer"],
            expected_filtered_answer,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluator.data["is_language_correct"],
            expected_is_lang_correct,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluator.data["is_answer_correct"],
            expected_is_answer_correct,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            correctness, expected_final_correctness, check_names=False
        )

    def test_get_data(self):
        evaluator = MCEvaluator(language_id=None)
        evaluator.get_correctness(self.sample_df.copy())
        retrieved_data = evaluator.get_data()
        self.assertIsInstance(retrieved_data, pd.DataFrame)
        self.assertTrue("correctness" in retrieved_data.columns)


class TestFGMathEvaluator(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "predictions": [
                    "The final answer is 123.",
                    "I think it's 456.",
                    "Result: 7890",
                ],
                "answer": ["123", "450", "7890"],
            }
        )
        self.lang_id_en = "en"

    @patch("mergenetic.evaluation.math_language.LanguageDetector")
    def test_init_with_language_id(self, mock_lang_detector_class):
        mock_detector_instance = MagicMock()
        mock_lang_detector_class.return_value = mock_detector_instance

        evaluator = FGMathEvaluator(language_id=self.lang_id_en)

        self.assertEqual(evaluator.language_id, self.lang_id_en)
        mock_lang_detector_class.assert_called_once_with([self.lang_id_en])
        self.assertEqual(evaluator.lang_detector, mock_detector_instance)

    def test_init_with_language_id_none(self):
        evaluator = FGMathEvaluator(language_id=None)
        self.assertIsNone(evaluator.language_id)
        # In FGMathEvaluator, lang_detector might not be explicitly set to None if language_id is None
        # The code is `if language_id: self.lang_detector = ...`
        # So, if language_id is None, self.lang_detector might not exist or be None.
        # Let's check it's not set to a LanguageDetector instance.
        self.assertFalse(
            hasattr(evaluator, "lang_detector") and evaluator.lang_detector is not None
        )

    def test_extract_answer_static_method(self):
        self.assertEqual(
            FGMathEvaluator._extract_answer("Answer is 123 then 456."), "456"
        )
        self.assertEqual(FGMathEvaluator._extract_answer("Only 789."), "789")
        self.assertEqual(FGMathEvaluator._extract_answer("No numbers here."), "None")
        self.assertEqual(
            FGMathEvaluator._extract_answer("Number 12.34 should be 56"), "56"
        )  # Extracts last sequence of digits
        self.assertEqual(FGMathEvaluator._extract_answer(""), "None")

    @patch.object(FGMathEvaluator, "_validate_dataframe")
    def test_get_correctness_no_language_id(self, mock_validate_df):
        evaluator = FGMathEvaluator(language_id=None)
        correctness = evaluator.get_correctness(self.sample_df.copy())

        mock_validate_df.assert_called_once()
        expected_filtered = pd.Series(
            ["123", "456", "7890"], name="predictions_filtered"
        )
        expected_correctness = pd.Series([True, False, True], name="correctness")

        pd.testing.assert_series_equal(
            evaluator.data["predictions_filtered"], expected_filtered, check_names=False
        )
        pd.testing.assert_series_equal(
            correctness, expected_correctness, check_names=False
        )
        self.assertTrue("evaluation_log" in evaluator.data.columns)

    @patch.object(FGMathEvaluator, "_validate_dataframe")
    @patch("mergenetic.evaluation.math_language.LanguageDetector")
    def test_get_correctness_with_language_id(
        self, mock_lang_detector_class, mock_validate_df
    ):
        mock_detector_instance = MagicMock()
        mock_detector_instance._get_language.side_effect = [
            "__label__en",
            "UNK",
            "__label__fr",
        ]  # fr is wrong
        mock_lang_detector_class.return_value = mock_detector_instance

        evaluator = FGMathEvaluator(language_id=self.lang_id_en)
        self.assertIsNotNone(evaluator.lang_detector)
        self.assertEqual(
            evaluator.lang_detector, mock_detector_instance
        )  # Verify mock is set

        correctness = evaluator.get_correctness(self.sample_df.copy())

        mock_validate_df.assert_called_once()
        self.assertEqual(
            mock_detector_instance._get_language.call_count, len(self.sample_df)
        )

        expected_filtered = pd.Series(
            ["123", "456", "7890"], name="predictions_filtered"
        )
        base_correctness = pd.Series([True, False, True])  # Answer correctness
        lang_correctness = pd.Series(
            [True, True, False]
        )  # Language correctness (en or UNK is True, fr is False)
        expected_final_correctness = base_correctness & lang_correctness
        # True & True = True
        # False & True = False
        # True & False = False

        pd.testing.assert_series_equal(
            evaluator.data["predictions_filtered"], expected_filtered, check_names=False
        )
        pd.testing.assert_series_equal(
            correctness, expected_final_correctness, check_names=False
        )
        self.assertTrue("evaluation_log" in evaluator.data.columns)
        self.assertTrue("language" in evaluator.data.columns)

    def test_get_data(self):
        evaluator = FGMathEvaluator(language_id=None)
        evaluator.get_correctness(self.sample_df.copy())
        retrieved_data = evaluator.get_data()
        self.assertIsInstance(retrieved_data, pd.DataFrame)
        self.assertTrue("correctness" in retrieved_data.columns)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
