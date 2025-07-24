#!/usr/bin/env python

import argparse
import logging
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mergenetic.cli")

# Define PROJECT_ROOT locally to avoid circular import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"Project root: {PROJECT_ROOT}")


def input_with_default(prompt_text, default=None):
    """Helper function to handle input with default values using prompt_toolkit."""
    input_history = InMemoryHistory()
    if default:
        text = prompt(f"{prompt_text} [{default}]: ", history=input_history)
        return text if text else default
    else:
        return prompt(f"{prompt_text}: ", history=input_history)


def yes_no_input(prompt_text, default="yes"):
    """Helper function to handle yes/no inputs using prompt_toolkit."""
    valid = {"yes": True, "y": True, "no": False, "n": False}
    input_history = InMemoryHistory()

    if default is None:
        prompt_str = f"{prompt_text} [y/n]: "
    elif default == "yes":
        prompt_str = f"{prompt_text} [Y/n]: "
    elif default == "no":
        prompt_str = f"{prompt_text} [y/N]: "

    while True:
        choice = prompt(prompt_str, history=input_history).lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            logger.warning("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def create_config_directory():
    """Create a directory for storing configuration files."""
    config_dir = PROJECT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mergenetic CLI for configuring model merging experiments."
    )
    parser.add_argument(
        "--eval-method",
        choices=["lm-eval", "custom"],
        help="Evaluation method: lm-eval (LM-Eval Harness) or custom",
        required=True,
    )
    parser.add_argument(
        "--merge-type",
        choices=["single", "multi"],
        help="Merging type: single (single language) or multi (multilingual)",
        required=True,
    )
    return parser.parse_args()


def handle_keyboard_interrupt(signum=None, frame=None):
    """Handle keyboard interrupt gracefully."""
    logger.warning("\n\nKeyboard interrupt received. Stopping execution...")
    logger.warning(
        "Exiting gracefully. Any saved configurations will remain available."
    )
    sys.exit(0)


def main():
    # Register signal handler for SIGINT (CTRL+C)
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    try:
        args = parse_arguments()

        logger.info("Welcome to Mergenetic CLI!")
        logger.info(
            "This tool will help you configure and run model merging experiments."
        )
        logger.info("-" * 50)

        # Step 1: Choose evaluation method
        if args.eval_method:
            use_lm_eval = args.eval_method == "lm-eval"
            logger.info(
                f"\nEvaluation method: {'LM-Eval Harness' if use_lm_eval else 'Custom evaluation'}"
            )
        else:
            logger.info("\nStep 1: Choose your evaluation method:")
            use_lm_eval = yes_no_input(
                "Do you want to use LM-Eval Harness for evaluation?"
            )

        # Step 2: Choose merging type
        if args.merge_type:
            is_multilingual = args.merge_type == "multi"
            logger.info(
                f"\nMerging type: {'Multilingual' if is_multilingual else 'Single language'}"
            )
        else:
            logger.info("\nStep 2: Choose your merging type:")
            is_multilingual = yes_no_input(
                "Do you want to perform multilingual merging?"
            )

        # Common configuration
        logger.info("\nStep 3: Basic Configuration")
        run_id = input_with_default("Enter a run ID", f"run_{uuid.uuid4().hex[:8]}")
        base_model = input_with_default(
            "Enter path to base model", "meta-llama/Llama-2-7b-hf"
        )

        # Device configuration
        device = input_with_default("Enter device (cuda, cpu, etc.)", "cuda")

        # Add dtype and task_type configuration options
        dtype = input_with_default(
            "Enter dtype (float16, float32, bfloat16)", "float16"
        )
        task_type = input_with_default(
            "Enter task type (FG_MATH, MULTIPLE_CHOICE)", "FG_MATH"
        )

        # Optimization parameters
        logger.info("\nStep 4: Optimization Parameters")
        pop_size = int(input_with_default("Population size", "10"))
        n_iter = int(input_with_default("Number of iterations", "10"))
        n_samples = int(input_with_default("Number of samples", "10"))
        eval_batch_size = int(input_with_default("Evaluation batch size", "32"))
        seed = int(input_with_default("Random seed", "42"))

        # Initialize config dictionary
        config = {
            "run_id": run_id,
            "base_model": base_model,
            "device": device,
            "pop_size": pop_size,
            "n_iter": n_iter,
            "n_samples": n_samples,
            "eval_batch_size": eval_batch_size,
            "seed": seed,
            "dtype": dtype,
            "task_type": task_type,
        }

        # Set up storage paths
        config_dir = create_config_directory()
        config["path_to_store_config"] = str(config_dir)
        config["path_to_store_merged_model"] = input_with_default(
            "Path to store merged model", str(PROJECT_ROOT / "models")
        )

        # Language-specific configuration
        if is_multilingual:
            logger.info("\nStep 5: Multilingual Configuration")
            n_langs = int(input_with_default("Number of languages", "2"))
            langs = []
            models = {}

            if use_lm_eval:
                tasks = {"search": {}, "test": {}}

                for i in range(n_langs):
                    lang = input_with_default(f"Language {i+1} ID", f"lang{i+1}")
                    langs.append(lang)
                    models[lang] = input_with_default(
                        f"Model path for {lang}", f"model_{lang}"
                    )
                    tasks["search"][lang] = input_with_default(
                        f"Search task for {lang}", f"task_{lang}"
                    )
                    tasks["test"][lang] = input_with_default(
                        f"Test task for {lang}", f"task_{lang}"
                    )

                config["langs"] = langs
                config["models"] = models
                config["tasks"] = tasks
                config["metric"] = input_with_default("Metric to use", "exact_match")
                config["bench"] = input_with_default("Benchmark type", "gsm8k")
                config["mode"] = input_with_default("Evaluation mode", "gmpirt")
                config["additional_templates_folder"] = input_with_default(
                    "Additional templates folder", "lm_tasks"
                )

            else:
                datasets = {}
                for i in range(n_langs):
                    lang = input_with_default(f"Language {i+1} ID", f"lang{i+1}")
                    langs.append(lang)
                    models[lang] = input_with_default(
                        f"Model path for {lang}", f"model_{lang}"
                    )
                    datasets[lang] = input_with_default(
                        f"Dataset path for {lang}", f"data_{lang}.csv"
                    )

                config["langs"] = langs
                config["models"] = models
                config["datasets"] = datasets
                config["bench"] = input_with_default("Benchmark type", "gsm8k")
                config["mode"] = input_with_default("Evaluation mode", "gmpirt")
        else:
            logger.info("\nStep 5: Single Language Configuration")
            lang = input_with_default("Language ID", "en")
            config["langs"] = [lang]
            model_path = input_with_default(f"Model path for {lang}", f"model_{lang}")
            config["models"] = {lang: model_path}

            if use_lm_eval:
                task_name = input_with_default(f"Task name for {lang}", f"task_{lang}")
                config["tasks"] = {
                    "search": {lang: task_name},
                    "test": {lang: f"{task_name}"},
                }
                config["metric"] = input_with_default("Metric to use", "exact_match")
                config["bench"] = input_with_default("Benchmark type", "gsm8k")
                config["mode"] = input_with_default("Evaluation mode", "gmpirt")
                config["additional_templates_folder"] = input_with_default(
                    "Additional templates folder", "lm_tasks"
                )
            else:
                dataset_path = input_with_default(
                    f"Dataset path for {lang}", f"data_{lang}.csv"
                )
                config["datasets"] = {lang: dataset_path}
                config["bench"] = input_with_default("Benchmark type", "gsm8k")
                config["mode"] = input_with_default("Evaluation mode", "gmpirt")

        # Save configuration
        config_file = config_dir / f"{run_id}_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"\nConfiguration saved to {config_file}")

        # Improved experiment launch section
        logger.info("\n" + "-" * 50)
        logger.info("EXPERIMENT LAUNCH")
        logger.info("-" * 50)

        # Determine the script to use
        if use_lm_eval:
            if is_multilingual:
                script_path = (
                    PROJECT_ROOT
                    / "experiments"
                    / "evolutionary-merging-lm-harness"
                    / "end2end_multilingual.py"
                )
                script_type = "LM-Eval Harness (Multilingual)"
            else:
                script_path = (
                    PROJECT_ROOT
                    / "experiments"
                    / "evolutionary-merging-lm-harness"
                    / "end2end.py"
                )
                script_type = "LM-Eval Harness (Single Language)"
        else:
            if is_multilingual:
                script_path = (
                    PROJECT_ROOT
                    / "experiments"
                    / "evolutionary-merging"
                    / "end2end_multilingual.py"
                )
                script_type = "Custom Evaluation (Multilingual)"
            else:
                script_path = (
                    PROJECT_ROOT / "experiments" / "evolutionary-merging" / "end2end.py"
                )
                script_type = "Custom Evaluation (Single Language)"

        # Display experiment summary
        logger.info(f"Experiment type: {script_type}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Configuration: {config_file}")
        logger.info(
            f"Command: python {script_path} --config {config_file} --run_id {run_id}\n"
        )

        # Ask for confirmation
        run_script = yes_no_input("Do you want to launch the experiment now?")

        if run_script:
            logger.info("\nLaunching experiment...")
            logger.info("This may take a while depending on your configuration.")
            logger.info("You can monitor the progress in the logs below:")
            logger.info(
                "(To stop the experiment, press CTRL+C once. The CLI will exit gracefully.)\n"
            )

            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--config",
                        str(config_file),
                        "--run_id",
                        run_id,
                    ],
                    check=True,
                )

                logger.info("\n" + "=" * 50)
                logger.info("✅ Experiment completed successfully!")
                logger.info(
                    f"Results are saved in the directory specified in your configuration."
                )
                logger.info("=" * 50)

            except subprocess.CalledProcessError as e:
                logger.error("\n" + "=" * 50)
                logger.error(f"❌ Error running experiment: {e}")
                logger.error("Check the logs above for more details.")
                logger.error("=" * 50)
                sys.exit(1)
            except KeyboardInterrupt:
                logger.warning("\n" + "=" * 50)
                logger.warning("⚠️ Experiment interrupted by user (CTRL+C)")
                logger.warning("The experiment process has been terminated.")
                logger.warning(
                    "Any partial results may be available in the output directory."
                )
                logger.warning("=" * 50)
                sys.exit(130)  # Standard exit code for SIGINT
        else:
            logger.info("\nExperiment not launched. To run it later, use this command:")
            logger.info(
                f"python {script_path} --config {config_file} --run_id {run_id}"
            )
            logger.info("\nGood luck with your experiment!")

    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
