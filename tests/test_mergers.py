import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from mergenetic.merging.dare_taskarithmetic_merger import DareTaskArithmeticMerger
from mergenetic.merging.linear_merger import LinearMerger
from mergenetic.merging.merger import Merger
from mergenetic.merging.slerp_merger import SlerpMerger
from mergenetic.merging.taskarithmetic_merger import TaskArithmeticMerger
from mergenetic.merging.ties_dare_merger import TiesDareMerger
from mergenetic.merging.ties_merger import TiesMerger

# Common parameters
RUN_ID = "test_run_001"
BASE_MODEL_PATH_STR = "meta-math/MetaMath-Mistral-7B"
MODEL_PATHS_MULTI_STR = [
    "OpenLLM-Ro/RoMistral-7b-Instruct",
    "DeepMount00/Mistral-Ita-7b",
]
MODEL_1_PATH_SLERP_STR = "OpenLLM-Ro/RoMistral-7b-Instruct"
DTYPE = "bfloat16"
LAYER_RANGE_SLERP = "[0,15]"


class BaseMergerTest(unittest.TestCase):
    def setUp(self):
        # Create real temporary directories for path management
        self.temp_yaml_dir = Path(tempfile.mkdtemp(prefix="mergenetic_test_yaml_"))
        self.temp_model_dir = Path(tempfile.mkdtemp(prefix="mergenetic_test_model_"))

        # Expected path for the YAML file based on Merger base class logic
        self.expected_yaml_file_path = self.temp_yaml_dir / "config.yaml"
        # Expected path for the merged model directory
        self.expected_merged_model_path = self.temp_model_dir / RUN_ID

    def tearDown(self):
        shutil.rmtree(self.temp_yaml_dir)
        shutil.rmtree(self.temp_model_dir)


class TestTiesMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = TiesMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,
            model_paths=MODEL_PATHS_MULTI_STR,
            path_to_store_yaml=str(self.temp_yaml_dir),  # Pass directory
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    def test_initialization(self):
        self.assertEqual(self.merger.path_to_base_model, Path(BASE_MODEL_PATH_STR))
        self.assertEqual(
            self.merger.model_paths, [Path(p) for p in MODEL_PATHS_MULTI_STR]
        )
        self.assertEqual(self.merger.dtype, DTYPE)
        # Base class attributes
        self.assertEqual(self.merger.path_to_store_yaml, self.expected_yaml_file_path)
        self.assertEqual(
            self.merger.path_to_store_merged_model, self.expected_merged_model_path
        )

    @patch("mergenetic.merging.ties_merger.yaml.dump")
    @patch("mergenetic.merging.ties_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mergenetic.merging.ties_merger.logger")
    def test_create_individual_configuration(
        self, mock_logger, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        weights_and_densities = [0.7, 0.3, 0.9, 0.6]  # w1, w2, d1, d2
        num_models_to_merge = len(MODEL_PATHS_MULTI_STR)

        returned_path = self.merger.create_individual_configuration(
            weights_and_densities
        )

        self.assertEqual(returned_path, self.expected_yaml_file_path)
        # Check that the parent directory of path_to_store_yaml was asked to be created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_open.assert_called_once_with(self.expected_yaml_file_path, "w")

        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]

        self.assertEqual(config_dumped["merge_method"], "ties")
        self.assertEqual(config_dumped["base_model"], BASE_MODEL_PATH_STR)
        self.assertEqual(config_dumped["dtype"], DTYPE)
        self.assertTrue(config_dumped["parameters"]["int8_mask"])

        self.assertEqual(len(config_dumped["models"]), num_models_to_merge + 1)
        self.assertEqual(config_dumped["models"][0]["model"], BASE_MODEL_PATH_STR)

        self.assertEqual(config_dumped["models"][1]["model"], MODEL_PATHS_MULTI_STR[0])
        self.assertEqual(
            config_dumped["models"][1]["parameters"]["weight"], weights_and_densities[0]
        )
        self.assertEqual(
            config_dumped["models"][1]["parameters"]["density"],
            weights_and_densities[num_models_to_merge],
        )

        mock_logger.info.assert_called_with(
            f"Configuration file created at {self.expected_yaml_file_path}"
        )


class TestTiesDareMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = TiesDareMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,
            model_paths=MODEL_PATHS_MULTI_STR,
            path_to_store_yaml=str(self.temp_yaml_dir),
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    @patch("mergenetic.merging.ties_dare_merger.yaml.dump")
    @patch("mergenetic.merging.ties_dare_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_individual_configuration(
        self, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        weights_and_densities = [0.7, 0.3, 0.9, 0.6]
        self.merger.create_individual_configuration(weights_and_densities)
        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]
        self.assertEqual(config_dumped["merge_method"], "dare_ties")
        # Other assertions similar to TiesMerger


class TestTaskArithmeticMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = TaskArithmeticMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,
            model_paths=MODEL_PATHS_MULTI_STR,
            path_to_store_yaml=str(self.temp_yaml_dir),
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    @patch("mergenetic.merging.taskarithmetic_merger.yaml.dump")
    @patch("mergenetic.merging.taskarithmetic_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_individual_configuration(
        self, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        weights = [0.6, 0.4]
        num_models_to_merge = len(MODEL_PATHS_MULTI_STR)
        self.merger.create_individual_configuration(weights)
        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]
        self.assertEqual(config_dumped["merge_method"], "task_arithmetic")
        self.assertEqual(config_dumped["base_model"], BASE_MODEL_PATH_STR)
        self.assertEqual(len(config_dumped["models"]), num_models_to_merge + 1)
        self.assertEqual(config_dumped["models"][1]["parameters"]["weight"], weights[0])
        self.assertNotIn("density", config_dumped["models"][1]["parameters"])


class TestDareTaskArithmeticMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = DareTaskArithmeticMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,
            model_paths=MODEL_PATHS_MULTI_STR,
            path_to_store_yaml=str(self.temp_yaml_dir),
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    @patch("mergenetic.merging.dare_taskarithmetic_merger.yaml.dump")
    @patch("mergenetic.merging.dare_taskarithmetic_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_individual_configuration(
        self, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        weights = [0.6, 0.4]
        self.merger.create_individual_configuration(weights)
        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]
        self.assertEqual(
            config_dumped["merge_method"], "dare_linear"
        )  # Code uses dare_linear
        self.assertEqual(config_dumped["base_model"], BASE_MODEL_PATH_STR)


class TestLinearMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = LinearMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,  # Passed but not used in config
            model_paths=MODEL_PATHS_MULTI_STR,
            path_to_store_yaml=str(self.temp_yaml_dir),
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    @patch("mergenetic.merging.linear_merger.yaml.dump")
    @patch("mergenetic.merging.linear_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_individual_configuration(
        self, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        weights = [0.6, 0.4]
        num_models_to_merge = len(MODEL_PATHS_MULTI_STR)
        self.merger.create_individual_configuration(weights)
        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]
        self.assertEqual(config_dumped["merge_method"], "linear")
        self.assertNotIn("base_model", config_dumped)
        self.assertEqual(len(config_dumped["models"]), num_models_to_merge)
        self.assertEqual(config_dumped["models"][0]["model"], MODEL_PATHS_MULTI_STR[0])


class TestSlerpMerger(BaseMergerTest):
    def setUp(self):
        super().setUp()
        self.merger = SlerpMerger(
            run_id=RUN_ID,
            path_to_base_model=BASE_MODEL_PATH_STR,
            layer_range_base_model=LAYER_RANGE_SLERP,
            path_to_model_1=MODEL_1_PATH_SLERP_STR,
            layer_range_model_1=LAYER_RANGE_SLERP,
            path_to_store_yaml=str(self.temp_yaml_dir),  # Pass directory
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    def test_initialization(self):
        self.assertEqual(self.merger.path_to_base_model, Path(BASE_MODEL_PATH_STR))
        self.assertEqual(self.merger.path_to_model_1, Path(MODEL_1_PATH_SLERP_STR))
        self.assertEqual(self.merger.layer_range_base_model, LAYER_RANGE_SLERP)
        self.assertEqual(self.merger.layer_range_model_1, LAYER_RANGE_SLERP)
        self.assertEqual(self.merger.path_to_store_yaml, self.expected_yaml_file_path)

    @patch("mergenetic.merging.slerp_merger.yaml.dump")
    @patch("mergenetic.merging.slerp_merger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mergenetic.merging.slerp_merger.logger")
    def test_create_individual_configuration(
        self, mock_logger, mock_file_open, mock_mkdir, mock_yaml_dump
    ):
        x_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.55]

        returned_path = self.merger.create_individual_configuration(x_params)
        self.assertEqual(returned_path, self.expected_yaml_file_path)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_open.assert_called_once_with(self.expected_yaml_file_path, "w")

        args, _ = mock_yaml_dump.call_args
        config_dumped = args[0]
        self.assertEqual(config_dumped["merge_method"], "slerp")
        self.assertEqual(config_dumped["base_model"], BASE_MODEL_PATH_STR)
        # ... detailed checks for Slerp t parameters and slices ...
        t_params_config = config_dumped["parameters"]["t"]
        self.assertEqual(t_params_config[0]["value"], x_params[0:5])  # self_attn
        self.assertEqual(t_params_config[1]["value"], x_params[5:10])  # mlp
        self.assertEqual(t_params_config[2]["value"], x_params[10])  # other


class TestMergerBaseFunctionality(BaseMergerTest):
    def setUp(self):
        super().setUp()

        class DummyMerger(Merger):  # Dummy concrete class for testing base Merger
            def create_individual_configuration(self, *args, **kwargs):
                return Path("dummy.yaml")

        self.merger_instance = DummyMerger(
            run_id=RUN_ID,
            path_to_store_yaml=str(self.temp_yaml_dir),
            path_to_store_merged_model=str(self.temp_model_dir),
            dtype=DTYPE,
        )

    @patch("mergenetic.merging.merger.logger")
    def test_check_and_delete_yaml_exists(self, mock_logger):
        # Create the dummy YAML file for the test
        self.expected_yaml_file_path.touch()
        self.assertTrue(self.expected_yaml_file_path.exists())

        self.merger_instance.check_and_delete_yaml()

        self.assertFalse(self.expected_yaml_file_path.exists())
        mock_logger.info.assert_called_with(f"Deleted: {self.expected_yaml_file_path}")

    @patch("mergenetic.merging.merger.logger")
    def test_check_and_delete_yaml_not_exists(self, mock_logger):
        self.assertFalse(
            self.expected_yaml_file_path.exists()
        )  # Ensure it doesn't exist
        self.merger_instance.check_and_delete_yaml()
        mock_logger.info.assert_called_with(
            f"No file found at: {self.expected_yaml_file_path}"
        )

    @patch("mergenetic.merging.merger.logger")
    @patch("mergenetic.merging.merger.shutil.rmtree")
    def test_delete_merged_model_local_exists(self, mock_rmtree, mock_logger):
        # Ensure the directory exists for this test case
        self.expected_merged_model_path.mkdir(parents=True, exist_ok=True)
        self.assertTrue(
            self.expected_merged_model_path.exists(),
            "Test setup: Directory should exist",
        )

        self.merger_instance._delete_merged_model_local()  # Call the method under test

        mock_rmtree.assert_called_once_with(self.expected_merged_model_path)
        mock_logger.info.assert_called_with(
            f"Deleted folder and all contents: {self.expected_merged_model_path}"
        )
        # The actual directory will be cleaned up by tearDown if mock_rmtree doesn't delete it.

    @patch("mergenetic.merging.merger.logger")
    @patch("mergenetic.merging.merger.shutil.rmtree")
    def test_delete_merged_model_local_not_exists(self, mock_rmtree, mock_logger):
        # Ensure the directory does NOT exist for this test case
        # It should not exist by default if setUp doesn't create it or if a previous test cleaned it.
        if self.expected_merged_model_path.exists():
            shutil.rmtree(
                self.expected_merged_model_path
            )  # Explicitly remove if it exists
        self.assertFalse(
            self.expected_merged_model_path.exists(),
            "Test setup: Directory should not exist",
        )

        self.merger_instance._delete_merged_model_local()  # Call the method under test

        mock_rmtree.assert_not_called()
        mock_logger.info.assert_called_with(
            f"The folder does not exist: {self.expected_merged_model_path}"
        )

    @patch("mergenetic.merging.merger.run_merge")
    @patch("mergenetic.merging.merger.MergeConfiguration.model_validate")
    @patch("mergenetic.merging.merger.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(
        Merger, "_delete_merged_model_local"
    )  # Patching on the base class itself
    @patch("mergenetic.merging.merger.clean_gpu")
    @patch("mergenetic.merging.merger.torch.cuda.is_available", return_value=True)
    @patch("mergenetic.merging.merger.logger")
    def test_merge_model_from_configuration(
        self,
        mock_base_logger,
        mock_cuda_available,
        mock_clean_gpu,
        mock_delete_model,
        mock_file_open_read,
        mock_yaml_safe_load,
        mock_model_validate,
        mock_run_merge,
    ):
        mock_config_dict = {"merge_method": "dummy", "models": []}
        mock_yaml_safe_load.return_value = mock_config_dict
        mock_merge_config_instance = MagicMock(name="MergeConfigInstance")
        mock_model_validate.return_value = mock_merge_config_instance

        # Create a dummy yaml file for the read operation
        # self.expected_yaml_file_path.touch() # Not needed if open is fully mocked

        result_path = self.merger_instance.merge_model_from_configuration(
            self.expected_yaml_file_path
        )

        mock_clean_gpu.assert_any_call()
        self.assertEqual(mock_clean_gpu.call_count, 2)
        mock_delete_model.assert_called_once()  # Called on self.merger_instance
        mock_file_open_read.assert_called_once_with(
            self.expected_yaml_file_path, "r", encoding="utf-8"
        )
        mock_yaml_safe_load.assert_called_once_with(mock_file_open_read.return_value)
        mock_model_validate.assert_called_once_with(mock_config_dict)
        mock_cuda_available.assert_called_once()

        mock_run_merge.assert_called_once()
        args_run_merge, kwargs_run_merge = mock_run_merge.call_args
        self.assertEqual(args_run_merge[0], mock_merge_config_instance)
        self.assertEqual(args_run_merge[1], str(self.expected_merged_model_path))

        options = kwargs_run_merge["options"]
        self.assertTrue(options.cuda)  # From mock_cuda_available
        self.assertTrue(options.copy_tokenizer)
        self.assertTrue(options.lazy_unpickle)
        self.assertTrue(options.low_cpu_memory)

        self.assertEqual(result_path, self.expected_merged_model_path)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
