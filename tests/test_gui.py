import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from gui.gui import create_interface


class TestCreateInterface(unittest.TestCase):

    @patch("gui.gui.create_config_directory")
    @patch("gui.gui.list_configurations")
    @patch("gui.gui.get_lm_eval_tasks")
    def test_create_interface_returns_blocks_instance(
        self,
        mock_get_lm_eval_tasks,
        mock_list_configurations,
        mock_create_config_directory,
    ):
        """
        Test that create_interface runs and returns a gradio.Blocks instance.
        """
        # Configure mocks
        mock_get_lm_eval_tasks.return_value = ["task_a", "task_b", "gsm8k"]
        mock_list_configurations.return_value = ["config_x", "-- New Configuration --"]
        mock_create_config_directory.return_value = Path("mocked/config/dir")

        # Call the function to test
        interface = create_interface()

        # Assert that the returned object is an instance of gr.Blocks
        self.assertIsInstance(interface, gr.Blocks)

    @patch("gui.gui.create_config_directory")
    @patch("gui.gui.list_configurations")
    @patch("gui.gui.get_lm_eval_tasks")
    def test_create_interface_has_main_elements(
        self,
        mock_get_lm_eval_tasks,
        mock_list_configurations,
        mock_create_config_directory,
    ):
        """
        Test that the created interface contains essential high-level components
        like the title and main tabs.
        """
        # Configure mocks
        mock_get_lm_eval_tasks.return_value = [
            "task_a",
            "task_b",
            "gsm8k",
        ]  # gsm8k for default values
        mock_list_configurations.return_value = ["config_x", "-- New Configuration --"]
        mock_create_config_directory.return_value = Path("mocked/config/dir")

        # Call the function to test
        interface = create_interface()

        found_title_markdown = False
        found_tabs_component = False
        found_config_tab_item = False
        found_execution_tab_item = False

        # Iterate through all blocks created within the interface
        # interface.blocks is a dictionary of all components in the Blocks instance
        for block_instance in interface.blocks.values():
            if isinstance(block_instance, gr.Markdown):
                if (
                    block_instance.value is not None
                    and block_instance.value.startswith(
                        "# Mergenetic: Evolutionary Model Merging"
                    )
                ):
                    found_title_markdown = True
            elif isinstance(block_instance, gr.Tabs):
                found_tabs_component = True
            elif isinstance(block_instance, gr.TabItem):
                if block_instance.label == "Configuration":
                    found_config_tab_item = True
                elif block_instance.label == "Execution":
                    found_execution_tab_item = True

        self.assertTrue(
            found_title_markdown, "Main title Markdown component not found."
        )
        self.assertTrue(found_tabs_component, "Gradio Tabs component not found.")
        self.assertTrue(found_config_tab_item, "Configuration TabItem not found.")
        self.assertTrue(found_execution_tab_item, "Execution TabItem not found.")


if __name__ == "__main__":
    unittest.main()
