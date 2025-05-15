import argparse
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from cli import mergenetic as cli_module


class BaseCLITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mergenetic_cli_test_"))

        self.mock_project_root = self.temp_dir / "mock_project_root"
        self.mock_project_root.mkdir(parents=True, exist_ok=True)

        (
            self.mock_project_root / "experiments" / "evolutionary-merging-lm-harness"
        ).mkdir(parents=True, exist_ok=True)
        (self.mock_project_root / "experiments" / "evolutionary-merging").mkdir(
            parents=True, exist_ok=True
        )
        (self.mock_project_root / "configs").mkdir(parents=True, exist_ok=True)

        self.project_root_patcher = patch.object(
            cli_module, "PROJECT_ROOT", self.mock_project_root
        )
        self.project_root_patcher.start()

        self.sys_exit_patcher = patch("sys.exit", autospec=True)
        self.mock_sys_exit = self.sys_exit_patcher.start()

        def exit_side_effect(status=0):
            raise SystemExit(status)

        self.mock_sys_exit.side_effect = exit_side_effect

        self.logger_patcher = patch.object(cli_module, "logger")
        self.mock_logger = self.logger_patcher.start()

        self.prompt_patcher = patch("cli.mergenetic.prompt")
        self.mock_prompt = self.prompt_patcher.start()

        self.history_patcher = patch("cli.mergenetic.InMemoryHistory")
        self.mock_in_memory_history_class = self.history_patcher.start()

        self.mock_in_memory_history_instance = MagicMock()
        self.mock_in_memory_history_class.return_value = (
            self.mock_in_memory_history_instance
        )

        if hasattr(cli_module, "input_history"):
            self.input_history_direct_patcher = patch.object(
                cli_module, "input_history", self.mock_in_memory_history_instance
            )
            self.input_history_direct_patcher.start()

    def tearDown(self):
        if hasattr(self, "input_history_direct_patcher"):
            self.input_history_direct_patcher.stop()
        self.history_patcher.stop()
        self.prompt_patcher.stop()
        self.logger_patcher.stop()
        self.sys_exit_patcher.stop()
        self.project_root_patcher.stop()
        shutil.rmtree(self.temp_dir)


class TestCLIHelpers(BaseCLITest):
    def test_input_with_default(self):
        self.mock_prompt.return_value = "user_value"
        result = cli_module.input_with_default("Enter value", "default_value")
        self.assertEqual(result, "user_value")
        self.mock_prompt.assert_called_with(
            "Enter value [default_value]: ",
            history=self.mock_in_memory_history_instance,
        )

        self.mock_prompt.return_value = ""
        result = cli_module.input_with_default("Enter value", "default_value")
        self.assertEqual(result, "default_value")
        self.mock_prompt.assert_called_with(
            "Enter value [default_value]: ",
            history=self.mock_in_memory_history_instance,
        )

        self.mock_prompt.return_value = "specific_value"
        result = cli_module.input_with_default("Enter specific value")
        self.assertEqual(result, "specific_value")
        self.mock_prompt.assert_called_with(
            "Enter specific value: ", history=self.mock_in_memory_history_instance
        )

    def test_yes_no_input(self):
        self.mock_prompt.return_value = "yes"
        self.assertTrue(cli_module.yes_no_input("Question?", default="no"))
        self.mock_prompt.assert_called_with(
            "Question? [y/N]: ", history=self.mock_in_memory_history_instance
        )

        self.mock_prompt.return_value = "n"
        self.assertFalse(cli_module.yes_no_input("Question?", default="yes"))
        self.mock_prompt.assert_called_with(
            "Question? [Y/n]: ", history=self.mock_in_memory_history_instance
        )

        self.mock_prompt.return_value = ""
        self.assertTrue(cli_module.yes_no_input("Question?", default="yes"))
        self.mock_prompt.assert_called_with(
            "Question? [Y/n]: ", history=self.mock_in_memory_history_instance
        )

        self.mock_prompt.return_value = ""
        self.assertFalse(cli_module.yes_no_input("Question?", default="no"))
        self.mock_prompt.assert_called_with(
            "Question? [y/N]: ", history=self.mock_in_memory_history_instance
        )

        self.mock_prompt.reset_mock()
        self.mock_logger.reset_mock()
        self.mock_prompt.side_effect = ["invalid", "y"]
        self.assertTrue(cli_module.yes_no_input("Question?", default="no"))
        self.assertEqual(self.mock_prompt.call_count, 2)
        self.mock_logger.warning.assert_called_with(
            "Please respond with 'yes' or 'no' (or 'y' or 'n')."
        )


class TestCLIArgParsing(BaseCLITest):
    def test_parse_arguments_valid(self):
        test_args = ["--eval-method", "lm-eval", "--merge-type", "single"]
        with patch.object(sys, "argv", ["mergenetic.py"] + test_args):
            args = cli_module.parse_arguments()
            self.assertEqual(args.eval_method, "lm-eval")
            self.assertEqual(args.merge_type, "single")

    def test_parse_arguments_missing_eval_method(self):
        test_args = ["--merge-type", "single"]
        with patch.object(sys, "argv", ["mergenetic.py"] + test_args):
            with self.assertRaises(SystemExit) as cm:
                cli_module.parse_arguments()
            self.assertEqual(cm.exception.code, 2)
        self.mock_sys_exit.assert_called_with(2)

    def test_parse_arguments_missing_merge_type(self):
        test_args = ["--eval-method", "lm-eval"]
        with patch.object(sys, "argv", ["mergenetic.py"] + test_args):
            with self.assertRaises(SystemExit) as cm:
                cli_module.parse_arguments()
            self.assertEqual(cm.exception.code, 2)
        self.mock_sys_exit.assert_called_with(2)


class TestCLIMainFlow(BaseCLITest):

    def _setup_common_mocks_for_main(self, user_inputs):
        self.mock_prompt.side_effect = user_inputs

        self.mock_file_open = mock_open()
        self.open_patcher = patch("builtins.open", self.mock_file_open)
        self.open_patcher.start()

        self.mock_yaml_dump = MagicMock()
        self.yaml_dump_patcher = patch("yaml.dump", self.mock_yaml_dump)
        self.yaml_dump_patcher.start()

        self.mock_subprocess_run = MagicMock()
        self.subprocess_run_patcher = patch("subprocess.run", self.mock_subprocess_run)
        self.subprocess_run_patcher.start()

        self.mock_signal_signal = MagicMock()
        self.signal_patcher = patch("signal.signal", self.mock_signal_signal)
        self.signal_patcher.start()

    def tearDown(self):
        if hasattr(self, "open_patcher"):
            self.open_patcher.stop()
        if hasattr(self, "yaml_dump_patcher"):
            self.yaml_dump_patcher.stop()
        if hasattr(self, "subprocess_run_patcher"):
            self.subprocess_run_patcher.stop()
        if hasattr(self, "signal_patcher"):
            self.signal_patcher.stop()
        super().tearDown()

    def test_main_flow_args_provided_single_lm_eval_no_run(self):
        sim_args = argparse.Namespace(eval_method="lm-eval", merge_type="single")
        with patch.object(cli_module, "parse_arguments", return_value=sim_args):
            user_inputs = [
                "test_run_01",
                "base/model",
                "cpu",
                "float32",
                "MULTIPLE_CHOICE",
                "5",
                "5",
                "50",
                "16",
                "123",
                str(self.mock_project_root / "merged_models_output"),
                "en",
                "model/en",
                "task_en",
                "acc_norm",
                "mmlu",
                "mpirt",
                "custom_lm_tasks",
                "no",
            ]
            self._setup_common_mocks_for_main(user_inputs)
            cli_module.main()
            self.mock_file_open.assert_called_once_with(
                self.mock_project_root / "configs" / "test_run_01_config.yaml", "w"
            )
            self.mock_yaml_dump.assert_called_once()
            dumped_config = self.mock_yaml_dump.call_args[0][0]
            self.assertEqual(dumped_config["run_id"], "test_run_01")
            self.assertEqual(dumped_config["base_model"], "base/model")
            self.assertTrue(dumped_config["tasks"]["search"]["en"] == "task_en")
            self.assertEqual(
                dumped_config["additional_templates_folder"], "custom_lm_tasks"
            )
            self.mock_subprocess_run.assert_not_called()
            self.mock_logger.info.assert_any_call(
                "\nExperiment not launched. To run it later, use this command:"
            )

    def test_main_flow_interactive_multi_custom_run_success(self):
        sim_args = argparse.Namespace(eval_method=None, merge_type=None)
        with patch.object(cli_module, "parse_arguments", return_value=sim_args):
            user_inputs = [
                "no",
                "yes",
                "multi_custom_run",
                "another/base",
                "cuda:0",
                "bfloat16",
                "FG_MATH",
                "20",
                "30",
                "100",
                "8",
                "2024",
                str(self.mock_project_root / "custom_merged_output"),
                "2",
                "de",
                "model/de",
                "data/de.csv",
                "es",
                "model/es",
                "data/es.csv",
                "custom_bench",
                "weighted",
                "yes",
            ]
            self._setup_common_mocks_for_main(user_inputs)
            self.mock_subprocess_run.return_value = MagicMock(
                check_returncode=lambda: None
            )
            cli_module.main()
            self.mock_file_open.assert_called_once_with(
                self.mock_project_root / "configs" / "multi_custom_run_config.yaml", "w"
            )
            dumped_config = self.mock_yaml_dump.call_args[0][0]
            self.assertEqual(dumped_config["langs"], ["de", "es"])
            self.assertEqual(dumped_config["datasets"]["de"], "data/de.csv")
            self.assertEqual(dumped_config["bench"], "custom_bench")
            self.mock_subprocess_run.assert_called_once()
            args_call = self.mock_subprocess_run.call_args[0][0]
            self.assertIn("end2end_multilingual.py", str(args_call[1]))
            self.assertIn("--config", args_call)
            self.assertIn("multi_custom_run_config.yaml", str(args_call[3]))
            self.mock_logger.info.assert_any_call(
                "✅ Experiment completed successfully!"
            )

    def test_main_flow_run_script_fails(self):
        sim_args = argparse.Namespace(eval_method="custom", merge_type="single")
        with patch.object(cli_module, "parse_arguments", return_value=sim_args):
            user_inputs = [
                "fail_run",
                "base/model",
                "cpu",
                "float16",
                "FG_MATH",
                "2",
                "2",
                "2",
                "2",
                "1",
                str(self.mock_project_root / "merged_output_fail"),
                "fr",
                "model/fr",
                "data/fr.csv",
                "bench_fr",
                "mean",
                "yes",
            ]
            self._setup_common_mocks_for_main(user_inputs)
            self.mock_subprocess_run.side_effect = subprocess.CalledProcessError(
                1, "cmd"
            )

            with self.assertRaises(SystemExit) as cm:
                cli_module.main()
            self.assertEqual(cm.exception.code, 1)

            self.mock_subprocess_run.assert_called_once()
            error_call_found_correctly = False
            for call_args_tuple in self.mock_logger.error.call_args_list:
                args, kwargs = call_args_tuple
                if args and isinstance(args[0], str):
                    if not kwargs.get("exc_info", False):
                        error_call_found_correctly = True
                        break
            self.assertTrue(
                error_call_found_correctly, "logger.error was not called appropriately"
            )
            self.mock_sys_exit.assert_called_with(1)

    def test_handle_keyboard_interrupt_in_main(self):
        sim_args = argparse.Namespace(eval_method=None, merge_type=None)
        with patch.object(cli_module, "parse_arguments", return_value=sim_args):
            self._setup_common_mocks_for_main([])
            self.mock_prompt.side_effect = KeyboardInterrupt

            with self.assertRaises(SystemExit) as cm:
                cli_module.main()
            self.assertEqual(cm.exception.code, 0)

            self.mock_logger.warning.assert_any_call(
                "\n\nKeyboard interrupt received. Stopping execution..."
            )
            self.mock_sys_exit.assert_called_with(0)

    def test_handle_keyboard_interrupt_during_subprocess(self):
        sim_args = argparse.Namespace(eval_method="custom", merge_type="single")
        with patch.object(cli_module, "parse_arguments", return_value=sim_args):
            user_inputs = [
                "kb_interrupt_run",
                "base/model",
                "cpu",
                "float16",
                "FG_MATH",
                "2",
                "2",
                "2",
                "2",
                "1",
                str(self.mock_project_root / "merged_interrupt"),
                "it",
                "model/it",
                "data/it.csv",
                "bench_it",
                "mean",
                "yes",
            ]
            self._setup_common_mocks_for_main(user_inputs)
            self.mock_subprocess_run.side_effect = KeyboardInterrupt

            with self.assertRaises(SystemExit) as cm:
                cli_module.main()
            self.assertEqual(cm.exception.code, 130)

            self.mock_subprocess_run.assert_called_once()
            self.mock_logger.warning.assert_any_call(
                "⚠️ Experiment interrupted by user (CTRL+C)"
            )
            self.mock_sys_exit.assert_called_with(130)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
