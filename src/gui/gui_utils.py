import importlib.util
import logging
import os
import queue
import subprocess
import sys
from pathlib import Path

import psutil
import yaml

# Configure specific loggers
logger = logging.getLogger("mergenetic.utils")
# Set httpx logger to WARNING to avoid the HTTP request info messages
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Define PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root: {PROJECT_ROOT}")

# Queue for log messages
log_queue = queue.Queue()


# Function to get component values
def get_component_value(component):
    """Extract value from a component or return the value itself if not a component."""
    if hasattr(component, "value"):
        return component.value
    return component


# Create a directory for storing configuration files
def create_config_directory():
    """Create a directory for storing configuration files."""
    config_dir = PROJECT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# Function to list available configurations
def list_configurations():
    """List all configuration files in the configs directory."""
    config_dir = PROJECT_ROOT / "configs"
    if not config_dir.exists():
        return []

    configs = list(config_dir.glob("*_config.yaml"))
    # Extract just the run_id from filenames and sort by modification time (newest first)
    config_options = []
    for config_file in sorted(configs, key=lambda x: x.stat().st_mtime, reverse=True):
        run_id = config_file.stem.split("_config")[0]
        config_options.append(f"{run_id} ({config_file.name})")

    return ["-- New Configuration --"] + config_options


# Function to load a configuration file
def load_configuration(config_selection):
    """Load a configuration file and return its contents."""
    if config_selection == "-- New Configuration --":
        return None

    # Extract the filename from the selection
    config_file = config_selection.split(" (")[1].rstrip(")")
    config_path = PROJECT_ROOT / "configs" / config_file

    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config, str(config_path)


# Global variable to store the current experiment process
current_experiment_process = None


# Function to run the experiment and update logs
def run_experiment(script_path, config_file, run_id):
    global current_experiment_process

    try:
        # Create process
        process = subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                "--config",
                str(config_file),
                "--run_id",
                run_id,
            ],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            preexec_fn=os.setsid,
            close_fds=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",  # Force stdout and stderr to be unbuffered
            },
            # Redirect stdout and stderr to the same stream
            text=True,
            bufsize=1,
        )

        # Store the process globally so we can terminate it later
        current_experiment_process = process

        log_output = []

        # Read output line by line
        for line in iter(process.stdout.readline, ""):
            log_output.append(line)
            yield "\n".join(log_output)

        process.stdout.close()
        return_code = process.wait()

        # Clear the current process reference
        current_experiment_process = None

        if return_code == 0:
            log_output.append("\n✅ Experiment completed successfully!")
        else:
            log_output.append(f"\n❌ Experiment failed with return code {return_code}")

        yield "\n".join(log_output)

    except Exception as e:
        current_experiment_process = None
        yield f"\n❌ Error running experiment: {e}"


# Function to get available tasks from lm_eval
def get_lm_eval_tasks():
    """Get the list of available tasks from lm_eval."""
    try:
        # Check if lm-evaluation-harness is installed
        if importlib.util.find_spec("lm_eval"):
            from lm_eval.tasks import TaskManager

            # Initialize the TaskManager and get available tasks
            task_manager = TaskManager()
            task_manager.initialize_tasks()
            available_tasks = task_manager.all_tasks
            available_tasks.sort()
            logger.info(f"Loaded {len(available_tasks)} tasks from lm_eval")
            return available_tasks
        else:
            logger.warning("lm_eval module not found, using placeholder tasks")
            # Provide some common tasks as placeholders
            return [
                "gsm8k",
                "mmlu",
                "hellaswag",
                "winogrande",
                "arc_easy",
                "arc_challenge",
            ]
    except Exception as e:
        logger.error(f"Error loading lm_eval tasks: {e}")
        return ["gsm8k", "mmlu", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]


# Function to stop a running experiment
def stop_experiment():
    global current_experiment_process

    if not current_experiment_process:
        return "No running experiment", "There is no experiment currently running."

    try:
        # Get the process ID
        pid = current_experiment_process.pid

        # Kill the Python process and its children
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Kill the main process
        current_experiment_process.kill()

        # Wait for the process to terminate
        current_experiment_process.wait()

        # Clear the global reference
        current_experiment_process = None

        return "Experiment stopped", "⚠️ Experiment was manually stopped."
    except Exception as e:
        current_experiment_process = None
        return (
            "Error stopping experiment",
            f"⚠️ Error when stopping experiment: {str(e)}",
        )
