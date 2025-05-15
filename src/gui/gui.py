import logging
import uuid
from pathlib import Path

import gradio as gr
import yaml

# Import utility functions from gui_utils.py
from gui.gui_utils import (
    PROJECT_ROOT,
    create_config_directory,
    get_component_value,
    get_lm_eval_tasks,
    list_configurations,
    load_configuration,
    run_experiment,
    stop_experiment,
)

# Define component mappings for configuration
COMPONENT_TYPES = ["lang", "model", "task", "dataset"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Configure specific loggers
logger = logging.getLogger("mergenetic.gui")
# Set httpx logger to WARNING to avoid the HTTP request info messages
logging.getLogger("httpx").setLevel(logging.WARNING)
# Also silence other potentially verbose loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Function to apply loaded configuration to UI components
def apply_configuration(
    config,
    eval_method,
    merge_type,
    run_id,
    base_model,
    device,
    dtype,
    pop_size,
    n_iter,
    n_samples,
    eval_batch_size,
    seed,
    path_to_store_config,
    path_to_store_merged_model,
    n_langs,
    lang_single,
    model_path_single,
    task_name_single,
    metric_single,
    additional_templates_folder_single,
    bench,
    mode,
    *lang_components,
):
    """Apply loaded configuration to UI components."""
    if not config:
        # Generate a new run_id for a fresh configuration
        new_run_id = f"run_{uuid.uuid4().hex[:8]}"
        return (
            # Default values for a new configuration
            eval_method.value,  # Keep current evaluation method
            merge_type.value,  # Keep current merge type
            new_run_id,
            base_model.value,
            device.value,
            dtype.value,
            pop_size.value,
            n_iter.value,
            n_samples.value,
            eval_batch_size.value,
            seed.value,
            path_to_store_config.value,
            path_to_store_merged_model.value,
            n_langs.value,
            lang_single.value,
            model_path_single.value,
            task_name_single.value,
            metric_single.value,
            additional_templates_folder_single.value,
            bench.value,
            mode.value,
        ) + lang_components

    # Force lm-eval
    use_lm_eval = True
    # Determine if the config is for multilingual or single language
    is_multilingual = len(config.get("langs", [])) > 1

    # Basic fields
    updated_fields = [
        "lm-eval",
        "multi" if is_multilingual else "single",
        config.get("run_id", run_id.value),
        config.get("base_model", base_model.value),
        config.get("device", device.value),
        config.get("dtype", dtype.value),
        config.get("pop_size", pop_size.value),
        config.get("n_iter", n_iter.value),
        config.get("n_samples", n_samples.value),
        config.get("eval_batch_size", eval_batch_size.value),
        config.get("seed", seed.value),
        config.get("path_to_store_config", path_to_store_config.value),
        config.get("path_to_store_merged_model", path_to_store_merged_model.value),
    ]

    # Language configuration
    if is_multilingual:
        langs = config.get("langs", [])
        updated_fields.append(len(langs))  # n_langs value

        # Default values for single language fields
        updated_fields.extend(
            [
                lang_single.value,
                model_path_single.value,
                task_name_single.value,
                config.get("metric", metric_single.value),
                config.get(
                    "additional_templates_folder",
                    additional_templates_folder_single.value,
                ),
            ]
        )
    else:
        # Single language configuration
        updated_fields.extend(
            [
                2,  # Default n_langs for multilingual (won't be used in single mode)
                "",  # Always use empty string for lang_single
                config.get("models", {}).get("task0", model_path_single.value),
                config.get("tasks", {})
                .get("search", {})
                .get("task0", task_name_single.value),
                config.get("metric", metric_single.value),
                config.get(
                    "additional_templates_folder",
                    additional_templates_folder_single.value,
                ),
            ]
        )

    # Common fields
    updated_fields.extend(
        [
            config.get("bench", bench.value),
            config.get("mode", mode.value),
        ]
    )

    # Handle multilingual language components
    if is_multilingual:
        lang_vals = lang_components[:5]
        model_vals = lang_components[5:10]
        task_vals = lang_components[10:15]

        langs = config.get("langs", [])
        models = config.get("models", {})

        # Update multilingual components
        for i in range(5):  # We support up to 5 languages
            if i < len(langs):
                lang_id = langs[i]
                updated_fields.append(lang_id)  # Language ID
                updated_fields.append(
                    models.get(lang_id, model_vals[i].value)
                )  # Model path

                # Tasks for LM-Eval
                task = (
                    config.get("tasks", {})
                    .get("search", {})
                    .get(lang_id, task_vals[i].value)
                )
                updated_fields.append(task)
            else:
                # Default values for unused language slots
                updated_fields.append(lang_vals[i].value)
                updated_fields.append(model_vals[i].value)
                updated_fields.append(task_vals[i].value)
    else:
        # For single language config, keep the default values for multilingual components
        updated_fields.extend(lang_components)

    # Also return the configuration YAML and file path
    yaml_str = yaml.dump(config, default_flow_style=False) if config else ""
    config_file_path = (
        config.get("path_to_store_config", path_to_store_config.value)
        + "/"
        + config.get("run_id", "unknown")
        + "_config.yaml"
        if config
        else ""
    )

    return tuple(updated_fields) + (yaml_str, config_file_path)


# Completely refactor the load_and_apply_configuration function for more direct component updates:
def load_and_apply_configuration(
    config_selection,
    eval_method,
    merge_type,
    run_id,
    base_model,
    device,
    dtype,
    pop_size,
    n_iter,
    n_samples,
    eval_batch_size,
    seed,
    path_to_store_config,
    path_to_store_merged_model,
    n_langs,
    lang_single,
    model_path_single,
    task_name_single,
    metric_single,
    additional_templates_folder_single,
    bench,
    mode,
    metric,
    *lang_components,
):
    """Load a configuration file and apply it to the UI components."""
    # First load the configuration
    config_data = load_configuration(config_selection)

    if not config_data:
        # Handle new configuration case - return defaults with message
        new_run_id = f"run_{uuid.uuid4().hex[:8]}"
        message = f'<div style="padding: 0.5rem; background-color: #f0f4c3; border-radius: 0.5rem; margin: 0.5rem 0;"><span style="color: #33691e; font-weight: bold;">Started new configuration with run_id:</span> <span style="color: #33691e; font-weight: bold;">{new_run_id}</span></div>'
        return (
            (
                eval_method,
                merge_type,
                new_run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                path_to_store_config,
                path_to_store_merged_model,
                n_langs,
                lang_single,
                model_path_single,
                task_name_single,
                metric_single,
                additional_templates_folder_single,
                bench,
                mode,
                metric,
            )
            + lang_components
            + ("", "", message)
        )

    config = config_data[0]  # Extract the config dict from the tuple
    config_file_path = config_data[1]  # Extract the file path

    # Log for full transparency
    logger.info(f"LOADED CONFIG: {config}")

    # Extract the key configuration elements
    run_id_val = config.get("run_id", "unknown")
    base_model_val = config.get("base_model", "")
    tasks_dict = config.get("tasks", {}).get("search", {})
    models_dict = config.get("models", {})

    # Count the number of tasks
    num_tasks = len(tasks_dict)

    # Logging for diagnostics
    logger.info(f"Loaded configuration with {num_tasks} tasks")
    logger.info(f"Models: {models_dict}")
    logger.info(f"Tasks: {tasks_dict}")

    # Check if we're dealing with multi-task configuration
    is_multilingual = num_tasks > 1

    # Calculate the total number of components
    # We have lang_components = 15 total components (5 languages × 3 component types)
    # First 5 are language IDs, next 5 are model paths, last 5 are tasks
    total_components = len(lang_components)
    langs_per_type = 5  # We support up to 5 languages in the UI

    # Create result lists with original values as defaults
    updated_fields = []

    # Step 1: Start with basic fields
    updated_fields.extend(
        [
            "lm-eval",  # Evaluation method - always lm-eval in this version
            "multi" if is_multilingual else "single",  # Objective type
            run_id_val,  # Run ID
            config.get("base_model", base_model),  # Base model
            config.get("device", device),  # Device
            config.get("dtype", dtype),  # Data type
            config.get("pop_size", pop_size),  # Population size
            config.get("n_iter", n_iter),  # Number of iterations
            config.get("n_samples", n_samples),  # Number of samples
            config.get("eval_batch_size", eval_batch_size),  # Evaluation batch size
            config.get("seed", seed),  # Random seed
            config.get("path_to_store_config", path_to_store_config),  # Config path
            config.get(
                "path_to_store_merged_model", path_to_store_merged_model
            ),  # Model path
        ]
    )

    # Step 2: Add task-specific configurations
    if is_multilingual:
        # For multi-task mode
        updated_fields.append(num_tasks)  # Number of languages/tasks

        # Single language fields (not used in multi mode, but need to be set)
        updated_fields.extend(
            [
                "",  # Language ID (unused)
                model_path_single,  # Model path (unused)
                task_name_single,  # Task name (unused)
                config.get("metric", metric),  # Metric
                config.get(
                    "additional_templates_folder", additional_templates_folder_single
                ),  # Templates folder
            ]
        )
    else:
        # For single-task mode
        langs = config.get("langs", [""])  # Default to empty string if not specified
        first_lang = langs[0] if langs else ""

        # Use explicit model and task from config, or defaults
        single_model_path = models_dict.get(first_lang, model_path_single)
        single_task_name = tasks_dict.get(first_lang, task_name_single)

        logger.info(
            f"Single task - Lang: '{first_lang}', Model: {single_model_path}, Task: {single_task_name}"
        )

        updated_fields.extend(
            [
                2,  # Default n_langs value (not used for single mode)
                "",  # Always use empty string for lang_single
                single_model_path,  # Model path
                single_task_name,  # Task name
                config.get("metric", metric),  # Metric
                config.get(
                    "additional_templates_folder", additional_templates_folder_single
                ),  # Templates folder
            ]
        )

    # Step 3: Add global evaluation settings
    updated_fields.extend(
        [
            config.get("bench", bench),  # Benchmark
            config.get("mode", mode),  # Evaluation mode
            config.get(
                "metric", metric
            ),  # Metric (already included but adding for consistency)
        ]
    )

    # Step 4: Handle the multilanguage components
    # These are the arrays that contain:
    # - lang_inputs: 5 language ID text boxes
    # - model_inputs: 5 model path text boxes
    # - task_inputs: 5 task dropdown boxes

    # Create copies of the original component values to modify
    lang_values = list(lang_components[:langs_per_type])
    model_values = list(lang_components[langs_per_type : 2 * langs_per_type])
    task_values = list(lang_components[2 * langs_per_type : 3 * langs_per_type])

    # For debugging
    logger.info(f"Original lang values: {lang_values}")
    logger.info(f"Original model values: {model_values}")
    logger.info(f"Original task values: {task_values}")

    # Update with values from the config where available
    if is_multilingual:
        # For multi-task, update all components using taskN keys
        for i in range(min(num_tasks, langs_per_type)):
            task_key = f"task{i}"

            # Update language field to the task ID
            lang_values[i] = task_key

            # Update model path from models dict
            model_values[i] = models_dict.get(task_key, model_values[i])

            # Update task value from tasks dict
            task_values[i] = tasks_dict.get(task_key, task_values[i])

            # Log each assignment for debugging
            logger.info(
                f"Setting task {i}: ID={task_key}, Model={model_values[i]}, Task={task_values[i]}"
            )
    else:
        # For single task, only update the first component
        first_lang = config.get("langs", ["task0"])[0]

        # Only update the first slot
        lang_values[0] = first_lang
        model_values[0] = models_dict.get(first_lang, model_values[0])
        task_values[0] = tasks_dict.get(first_lang, task_values[0])

        logger.info(
            f"Single task update: Lang={lang_values[0]}, Model={model_values[0]}, Task={task_values[0]}"
        )

    # Add component values to updated_fields
    updated_fields.extend(lang_values + model_values + task_values)

    # Step 5: Generate YAML preview
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Create a more compact message format
    message = f"""
    <div style="padding: 0.5rem; background-color: #e8f5e9; border-radius: 0.5rem; margin: 0.25rem 0; border: 1px solid #81c784; font-size: 0.9rem;">
        <span style="color: #2e7d32; font-weight: bold;">✅ Configuration Loaded</span>
    </div>
    """

    # Return the updated values
    return tuple(updated_fields) + (yaml_str, config_file_path, message)


# Main function to create the Gradio interface
def create_interface():
    # Get available LM-Eval tasks
    lm_eval_tasks = get_lm_eval_tasks()

    # Define allowed benchmark types for non-mean modes
    allowed_benchmarks = [
        "arc",
        "gsm8k",
        "hellaswag",
        "mmlu",
        "truthfulqa",
        "winogrande",
    ]
    evaluation_modes = ["mean", "irt", "pirt", "mpirt", "gmpirt"]

    # Initialize input collection lists before UI components are created
    lang_inputs = []
    model_inputs = []
    task_inputs = []

    with gr.Blocks(
        title="Mergenetic GUI", css="h2, h3 { margin-top: 1rem; padding-top: 1rem; }"
    ) as interface:
        gr.Markdown("# Mergenetic: Evolutionary Model Merging")
        gr.Markdown(
            "Configure and run evolutionary model merging experiments with this visual interface."
        )

        # Add state to track if a configuration has been generated
        config_generated = gr.State(value=False)

        with gr.Tabs() as tabs:
            with gr.TabItem("Configuration", id="configuration_tab") as config_tab:
                # Add state variable for the current step
                current_step = gr.State(value=0)

                # Define function to generate progress bar HTML
                def generate_progress_bar(current_step, titles):
                    """Generate HTML for a visual progress bar."""
                    # Create a container that will fully enclose the steps and their labels
                    html = '<div style="position: relative; margin: 1.5rem 0; padding: 1.5rem 0.5rem 2rem 0.5rem; background-color: #f8f9fa; border-radius: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'

                    # Create a flex container for the steps inside the main container
                    html += '<div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative;">'

                    # Add connecting line behind the step indicators
                    html += '<div style="position: absolute; height: 3px; background-color: #e0e0e0; top: 18px; left: 2%; right: 2%; z-index: 0;"></div>'

                    # Generate step indicators
                    for i, title in enumerate(titles):
                        # Determine the status of this step
                        if i < current_step:  # Completed step
                            status_color = "#4CAF50"  # Green
                            text_color = "#4CAF50"
                            bg_color = "#E8F5E9"
                            border = "2px solid #4CAF50"
                            status_text = "✓"
                        elif i == current_step:  # Current step
                            status_color = "#1565C0"  # Blue
                            text_color = "#1565C0"
                            bg_color = "#E3F2FD"
                            border = "2px solid #1565C0"
                            status_text = str(i + 1)
                        else:  # Future step
                            status_color = "#9E9E9E"  # Gray
                            text_color = "#9E9E9E"
                            bg_color = "#FFFFFF"
                            border = "1px solid #9E9E9E"
                            status_text = str(i + 1)

                        # Create step indicator
                        html += f"""
                        <div style="display: flex; flex-direction: column; align-items: center; z-index: 1; flex: 1; max-width: 16%;">
                            <div style="width: 36px; height: 36px; border-radius: 50%; background-color: {bg_color}; 
                                 border: {border}; display: flex; justify-content: center; align-items: center;
                                 color: {status_color}; font-weight: bold;">{status_text}</div>
                            <div style="margin-top: 0.5rem; text-align: center; color: {text_color}; 
                                 font-size: 0.8rem; font-weight: {('bold' if i == current_step else 'normal')};">
                                {title}
                            </div>
                        </div>
                        """

                    html += "</div></div>"  # Close both containers
                    return html

                # Progress indicator with improved styling - initial state
                with gr.Row():
                    step_titles = [
                        "Load Configuration",
                        "Evaluation Method",
                        "Basic Configuration",
                        "Task Configuration",  # Renamed from Language Configuration
                        "Optimization Parameters",
                        "Generate Configuration",
                    ]
                    initial_progress_bar = generate_progress_bar(0, step_titles)
                    progress_bar = gr.HTML(value=initial_progress_bar)

                # Navigation buttons moved to the top with better spacing
                with gr.Row(equal_height=True, variant="panel"):
                    with gr.Column(scale=1, min_width=100):
                        prev_btn = gr.Button(
                            "← Previous", variant="secondary", visible=False, size="lg"
                        )
                    with gr.Column(scale=1, min_width=100):
                        next_btn = gr.Button("Next →", variant="primary", size="lg")

                # Load existing configuration section - Step 0
                with gr.Group(visible=True) as step_0_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Load Configuration</h2>'
                    )

                    # Put dropdown and refresh button in the same row
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=8):
                            config_dropdown = gr.Dropdown(
                                choices=list_configurations(),
                                label="Select Configuration",
                                info="Select a configuration to load or choose '--New Configuration--' to create a new one",
                                value="-- New Configuration --",
                            )
                        with gr.Column(scale=1, min_width=50):
                            refresh_btn = gr.Button(
                                "🔄", variant="secondary", min_width=50
                            )

                    # Add feedback message area
                    config_load_message = gr.HTML(
                        value="", elem_id="config_load_message"
                    )

                # Step 1: Evaluation Method - with improved title spacing
                with gr.Group(visible=False) as step_1_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Step 1: Evaluation Method</h2>'
                    )
                    eval_method = gr.Radio(
                        ["lm-eval"],  # Only lm-eval remains
                        label="Evaluation Method",
                        info="LM-Eval Harness",
                        value="lm-eval",
                        visible=False,  # Hide this input
                    )
                    mode = gr.Dropdown(
                        choices=evaluation_modes,
                        label="Evaluation Mode",
                        value="gmpirt",
                        info="Select the evaluation method",
                    )
                    bench = gr.Dropdown(
                        choices=allowed_benchmarks,
                        label="Benchmark Type",
                        value="gsm8k",
                        info="Select the benchmark type for ability estimation (disabled for mean mode)",
                    )
                    metric = gr.Textbox(
                        label="Evaluation Metric",
                        value="exact_match",
                        info="Metric to use for evaluation in lm-eval (e.g., exact_match, acc, acc_norm, f1)",
                    )

                # Step 2: Basic Configuration - with improved title spacing
                with gr.Group(visible=False) as step_2_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Step 2: Basic Configuration</h2>'
                    )
                    # Rename from "Merging Type" to "Objective Type"
                    merge_type = gr.Radio(
                        ["single", "multi"],
                        label="Objective Type",
                        info="Choose between single task optimization or multi-task optimization",
                        value="single",
                    )
                    run_id = gr.Textbox(
                        label="Run ID",
                        value=f"run_{uuid.uuid4().hex[:8]}",
                        info="Unique identifier for this experiment",
                    )
                    base_model = gr.Textbox(
                        label="Base Model Path",
                        value="mistralai/Mistral-7B-v0.1",
                        info="Path to the base model",
                    )
                    device = gr.Textbox(
                        label="Device",
                        value="cuda",
                        info="Device to use (cuda, cpu, etc.)",
                    )
                    dtype = gr.Dropdown(
                        ["float16", "float32", "bfloat16"],
                        label="Data Type",
                        value="float16",
                        info="Precision for model computations",
                    )

                # Step 3: Task Configuration (previously Language Configuration) - with improved title spacing
                with gr.Group(visible=False) as step_3_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Step 3: Task Configuration</h2>'
                    )

                    # Multilingual configuration
                    with gr.Group(
                        visible=False
                    ) as multi_group:  # <-- now hidden by default for "single" merging type
                        n_langs = gr.Number(
                            label="Number of Languages",
                            value=2,
                            minimum=1,
                            maximum=5,
                            precision=0,
                            info="Number of languages to configure",
                        )

                        # Hide the Task IDs by placing them in a group set to visible=False.
                        with gr.Group(visible=False) as task_ids_group:
                            gr.Markdown(
                                '<h3 style="margin: 0.5rem 0.5rem 0.5rem 0.5rem;">Task IDs</h3>'
                            )
                            lang_id_groups = []
                            for i in range(5):  # Support up to 5 tasks
                                with gr.Group(visible=False) as lang_id_group:
                                    lang = gr.Textbox(
                                        label=f"Task {i} ID",
                                        value=f"task{i}",
                                        info=f"Identifier for task {i}",
                                    )
                                    lang_inputs.append(lang)
                                lang_id_groups.append(lang_id_group)

                        # Model Paths Section - with improved heading
                        with gr.Group() as model_paths_section:
                            gr.Markdown(
                                '<h3 style="margin: 0.5rem 0.5rem 0.5rem 0.5rem;">Model Paths</h3>'
                            )
                            model_path_groups = []
                            for i in range(5):
                                with gr.Group(visible=i < 2) as model_path_group:
                                    model = gr.Textbox(
                                        label=f"Model Path for Task {i+1}",
                                        value=f"model_task{i+1}",
                                        info=f"Path to model for task {i+1}",
                                    )
                                    model_inputs.append(model)
                                model_path_groups.append(model_path_group)

                        # Tasks Section - with improved heading and dropdown selector
                        with gr.Group() as tasks_section:
                            gr.Markdown(
                                '<h3 style="margin: 0.5rem 0.5rem 0.5rem 0.5rem;">Tasks (for LM-Eval)</h3>'
                            )
                            task_groups = []
                            for i in range(5):
                                with gr.Group(visible=i < 2) as task_group:
                                    # Change the label format from "Task for Task {i+1}" to just "Task {i+1}"
                                    task = gr.Dropdown(
                                        choices=lm_eval_tasks,
                                        label=f"Task {i+1}",  # Simplified label
                                        value=(
                                            "gsm8k"
                                            if lm_eval_tasks
                                            and "gsm8k" in lm_eval_tasks
                                            else (
                                                lm_eval_tasks[0]
                                                if lm_eval_tasks
                                                else f"task_task{i+1}"
                                            )
                                        ),
                                        allow_custom_value=True,
                                        info=f"Task for task {i+1} (select from list or enter custom)",
                                    )
                                    task_inputs.append(task)
                                task_groups.append(task_group)

                            # Add Additional Tasks Folder field after the task dropdowns
                            gr.Markdown(
                                '<h3 style="margin: 0.5rem 0.5rem 0.5rem 0.5rem;">Additional Settings</h3>'
                            )
                            additional_templates_folder_multi = gr.Textbox(
                                label="Additional Tasks Folder",
                                value="lm_tasks",
                                info="Folder containing additional lm-eval tasks",
                            )

                    # Single language configuration
                    with gr.Group() as single_group:
                        # Keep the lang_single field but make it invisible
                        lang_single = gr.Textbox(
                            label="Language ID",
                            value="task0",  # Default to empty string
                            visible=False,  # Hide it from the UI
                            info="Identifier for the language",
                        )

                        model_path_single = gr.Textbox(
                            label="Model Path",
                            value="model_en",
                            info="Path to the model",
                        )

                        # LM-Eval specific inputs - with task dropdown
                        with gr.Group() as lm_eval_single:
                            # Replace task textbox with dropdown
                            task_name_single = gr.Dropdown(
                                choices=lm_eval_tasks,
                                label="Task Name",
                                value=(
                                    "gsm8k"
                                    if lm_eval_tasks and "gsm8k" in lm_eval_tasks
                                    else (
                                        lm_eval_tasks[0] if lm_eval_tasks else "task_en"
                                    )
                                ),
                                allow_custom_value=True,
                                info="Name of the task (select from list or enter custom)",
                            )
                            metric_single = gr.Textbox(
                                label="Metric",
                                value="exact_match",
                                visible=False,
                                info="Metric to use for evaluation",
                            )
                            additional_templates_folder_single = gr.Textbox(
                                label="Additional Tasks Folder",
                                value="lm_tasks",
                                info="Folder containing additional lm-eval tasks",
                            )

                    # Common settings now only contain the metric field that's used by both tabs
                    with gr.Group() as common_settings:
                        # Metric is already moved to Evaluation Method step, so we don't need anything here
                        pass

                # Step 4: Optimization Parameters (previously step 3) - with improved title spacing
                with gr.Group(visible=False) as step_4_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Step 4: Optimization Parameters</h2>'
                    )
                    pop_size = gr.Number(
                        label="Population Size",
                        value=10,
                        precision=0,
                        info="Size of population for evolutionary algorithm",
                    )
                    n_iter = gr.Number(
                        label="Number of Iterations",
                        value=10,
                        precision=0,
                        info="Number of iterations for the algorithm",
                    )
                    n_samples = gr.Number(
                        label="Number of Samples",
                        value=10,
                        precision=0,
                        info="Number of samples to use",
                    )
                    eval_batch_size = gr.Number(
                        label="Evaluation Batch Size",
                        value=32,
                        precision=0,
                        info="Batch size for evaluation",
                    )
                    seed = gr.Number(
                        label="Random Seed",
                        value=42,
                        precision=0,
                        info="Random seed for reproducibility",
                    )

                    # Storage Paths with improved spacing
                    gr.Markdown(
                        '<h3 style="margin: 0.5rem 0.5rem 0.5rem 0.5rem;">Storage Paths</h3>'
                    )
                    config_dir = create_config_directory()
                    path_to_store_config = gr.Textbox(
                        label="Path to Store Config",
                        value=str(config_dir),
                        info="Directory to store configuration files",
                    )
                    path_to_store_merged_model = gr.Textbox(
                        label="Path to Store Merged Model",
                        value=str(PROJECT_ROOT / "models"),
                        info="Directory to store merged model files",
                    )

                # Step 5: Generate Configuration - with modified buttons
                with gr.Group(visible=False) as step_5_group:
                    gr.Markdown(
                        '<h2 style="margin: 0.5rem 0.5rem;">Step 5: Generate Configuration</h2>'
                    )
                    with gr.Row():
                        generate_config_btn = gr.Button(
                            "Generate Configuration", variant="primary"
                        )
                        go_to_execution_btn = gr.Button(
                            "Go to Execution Tab",
                            variant="secondary",
                            interactive=False,  # Disabled by default
                        )

                    # Add a function to update the preview whenever inputs change
                    def update_preview(
                        eval_method_val,
                        merge_type_val,
                        run_id_val,
                        base_model_val,
                        device_val,
                        dtype_val,
                        pop_size_val,
                        n_iter_val,
                        n_samples_val,
                        eval_batch_size_val,
                        seed_val,
                        path_to_store_config_val,
                        path_to_store_merged_model_val,
                        n_langs_val,
                        lang_single_val,
                        model_path_single_val,
                        task_name_single_val,
                        metric_single_val,
                        additional_templates_folder_single_val,
                        bench_val,
                        mode_val,
                        metric_val,
                        *args,
                    ):
                        # Handle benchmark value for mean mode
                        if mode_val == "mean":
                            bench_val = ""  # Not used in mean mode

                        # Basic config structure
                        config = {
                            "run_id": run_id_val,
                            "base_model": base_model_val,
                            "device": device_val,
                            "dtype": dtype_val,
                            "pop_size": (
                                int(float(pop_size_val)) if pop_size_val else 10
                            ),
                            "n_iter": int(float(n_iter_val)) if n_iter_val else 10,
                            "n_samples": (
                                int(float(n_samples_val)) if n_samples_val else 10
                            ),
                            "eval_batch_size": (
                                int(float(eval_batch_size_val))
                                if eval_batch_size_val
                                else 32
                            ),
                            "seed": int(float(seed_val)) if seed_val else 42,
                            "path_to_store_config": path_to_store_config_val,
                            "path_to_store_merged_model": path_to_store_merged_model_val,
                            "mode": mode_val,
                            "metric": metric_val,
                        }

                        # Only include benchmark if it's not mean mode
                        if mode_val != "mean" and bench_val:
                            config["bench"] = bench_val

                        # Extract multi-language inputs
                        lang_vals = args[:5]
                        model_vals = args[5:10]
                        task_vals = args[10:15]

                        is_multilingual = merge_type_val == "multi"

                        if is_multilingual:
                            # For multi-task mode
                            actual_n_langs = (
                                int(float(n_langs_val)) if n_langs_val else 2
                            )
                            # Only include defined languages (based on n_langs)
                            langs = [lang_vals[i] for i in range(actual_n_langs)]
                            models = {
                                langs[i]: model_vals[i] for i in range(actual_n_langs)
                            }

                            config["langs"] = langs
                            config["models"] = models

                            # Create tasks dict
                            search_tasks = {
                                langs[i]: task_vals[i] for i in range(actual_n_langs)
                            }
                            test_tasks = {
                                k: v for k, v in search_tasks.items()
                            }  # Create a distinct copy

                            config["tasks"] = {
                                "search": search_tasks,
                                "test": test_tasks,
                            }
                            config["additional_templates_folder"] = (
                                additional_templates_folder_multi.value
                            )
                        else:
                            # For single-task mode - use "task0" as language ID
                            config["langs"] = ["task0"]
                            config["models"] = {"task0": model_path_single_val}

                            config["tasks"] = {
                                "search": {"task0": task_name_single_val},
                                "test": {"task0": task_name_single_val},
                            }
                            config["additional_templates_folder"] = (
                                additional_templates_folder_single_val
                            )

                        # Generate YAML preview
                        yaml_str = yaml.dump(
                            config, default_flow_style=False, sort_keys=False
                        )
                        return yaml_str

                    # Configuration output with reactive preview
                    config_output = gr.Textbox(
                        label="Configuration Preview", interactive=False, lines=10
                    )
                    config_file_path = gr.Textbox(
                        label="Configuration File Path", interactive=False
                    )

                # Hidden components for navigation direction
                next_value = gr.Number(value=1, visible=False)
                prev_value = gr.Number(value=-1, visible=False)

                # Updated collection of step groups for navigation (with reordered steps)
                step_groups = [
                    step_0_group,  # Step 0: Load Configuration
                    step_1_group,  # Step 1: Evaluation Method
                    step_2_group,  # Step 2: Basic Configuration
                    step_3_group,  # Step 3: Task Configuration (previously Language Configuration)
                    step_4_group,  # Step 4: Optimization Parameters (previously step 3)
                    step_5_group,  # Step 5: Generate Configuration
                ]

                # Updated step titles for progress indicator
                step_titles = [
                    "Load Configuration",
                    "Evaluation Method",
                    "Basic Configuration",
                    "Task Configuration",  # Renamed from Language Configuration
                    "Optimization Parameters",
                    "Generate Configuration",
                ]

                # Function to navigate between steps - using the new progress bar
                def navigate_step(direction, current):
                    new_step = current + direction
                    if new_step < 0:
                        new_step = 0
                    if new_step >= len(step_groups):
                        new_step = len(step_groups) - 1

                    # Update visibility of all step groups
                    visibility_updates = [
                        gr.update(visible=(i == new_step))
                        for i in range(len(step_groups))
                    ]

                    # Update navigation buttons visibility
                    prev_visible = new_step > 0
                    next_visible = new_step < len(step_groups) - 1

                    # Generate updated progress bar
                    progress_html = generate_progress_bar(new_step, step_titles)

                    return [
                        new_step,
                        progress_html,
                        gr.update(visible=prev_visible),
                        gr.update(visible=next_visible),
                    ] + visibility_updates

                # Connect navigation buttons using hidden number components
                next_btn.click(
                    navigate_step,
                    inputs=[next_value, current_step],
                    outputs=[current_step, progress_bar, prev_btn, next_btn]
                    + step_groups,
                )
                prev_btn.click(
                    navigate_step,
                    inputs=[prev_value, current_step],
                    outputs=[current_step, progress_bar, prev_btn, next_btn]
                    + step_groups,
                )

            with gr.TabItem("Execution", id="execution_tab") as execution_tab:
                gr.Markdown("## Experiment Execution")
                gr.Markdown(
                    "Launch your configured experiment and view logs in real-time."
                )

                with gr.Row():
                    launch_btn = gr.Button("Launch Experiment", variant="primary")
                    stop_btn = gr.Button("Stop Experiment", variant="stop")

                experiment_status = gr.Textbox(
                    label="Experiment Status", value="Not started", interactive=False
                )

                log_output = gr.TextArea(
                    label="Experiment Log", interactive=False, lines=20, autoscroll=True
                )

        # Function to update visibility of components based on selections
        def update_visibility(
            eval_method_val, merge_type_val, eval_mode_val, n_langs_val=2
        ):
            is_lm_eval = True  # Always lm-eval
            is_multi = merge_type_val == "multi"

            results = {
                "multi_group": gr.update(visible=is_multi),
                "single_group": gr.update(visible=not is_multi),
                "lm_eval_single": gr.update(visible=not is_multi and is_lm_eval),
                "tasks_section": gr.update(visible=is_multi and is_lm_eval),
            }

            # Update language group visibility for multilingual setup
            n_langs_val = (
                int(n_langs_val) if isinstance(n_langs_val, (int, float)) else 2
            )

            # Update each language component's visibility
            for i in range(5):
                results[f"lang_id_group_{i}"] = gr.update(visible=i < n_langs_val)
                results[f"model_path_group_{i}"] = gr.update(visible=i < n_langs_val)
                results[f"task_group_{i}"] = gr.update(
                    visible=is_lm_eval and i < n_langs_val
                )

            # Return updates in the correct order to match outputs
            return (
                [
                    results["multi_group"],
                    results["single_group"],
                    results["lm_eval_single"],
                    results["tasks_section"],
                ]
                + [gr.update(visible=i < n_langs_val) for i in range(5)]
                + [gr.update(visible=i < n_langs_val) for i in range(5)]
                + [gr.update(visible=is_lm_eval and i < n_langs_val) for i in range(5)]
            )

        # Connect visibility updates with proper component references
        eval_method.change(
            fn=update_visibility,
            inputs=[eval_method, merge_type, n_langs],
            outputs=[multi_group, single_group, lm_eval_single, tasks_section]
            + lang_id_groups
            + model_path_groups
            + task_groups,
        )

        merge_type.change(
            fn=update_visibility,
            inputs=[eval_method, merge_type, n_langs],
            outputs=[multi_group, single_group, lm_eval_single, tasks_section]
            + lang_id_groups
            + model_path_groups
            + task_groups,
        )

        n_langs.change(
            fn=update_visibility,
            inputs=[eval_method, merge_type, n_langs],
            outputs=[multi_group, single_group, lm_eval_single, tasks_section]
            + lang_id_groups
            + model_path_groups
            + task_groups,
        )

        # Function to update benchmark type visibility and choices based on evaluation mode
        def update_benchmark_visibility(mode_val):
            # update task inputs to only have allowed benchmarks if not in mean mode

            if mode_val == "mean":
                # For mean mode: disable the dropdown, set empty value, and include empty string option
                return gr.update(
                    interactive=False,
                    value="",
                    choices=[""],  # Include empty string as first option
                )
            else:
                # For non-mean modes: enable dropdown, keep gsm8k as default value
                return gr.update(
                    interactive=True,
                    value="gsm8k",
                    choices=allowed_benchmarks,  # Only allowed benchmarks for non-mean modes
                )

        # Connect mode change to benchmark visibility update
        mode.change(fn=update_benchmark_visibility, inputs=[mode], outputs=[bench])

        # Function to generate configuration - modified to update button state
        def generate_config(
            eval_method_val,
            merge_type_val,
            run_id_val,
            base_model_val,
            device_val,
            dtype_val,
            pop_size_val,
            n_iter_val,
            n_samples_val,
            eval_batch_size_val,
            seed_val,
            path_to_store_config_val,
            path_to_store_merged_model_val,
            n_langs_val,
            lang_single_val,
            model_path_single_val,
            task_name_single_val,
            metric_single_val,
            additional_templates_folder_single_val,
            bench_val,
            mode_val,
            metric_val,
            *args,
        ):
            # Handle benchmark value for mean mode
            if mode_val == "mean":
                bench_val = ""  # Not used in mean mode

            # Basic config common to all setups
            config = {
                "run_id": run_id_val,
                "base_model": base_model_val,
                "device": device_val,
                "dtype": dtype_val,
                "task_type": "lm_eval",  # Default task type
                "pop_size": int(pop_size_val),
                "n_iter": int(n_iter_val),
                "n_samples": int(n_samples_val),
                "eval_batch_size": int(eval_batch_size_val),
                "seed": int(seed_val),
                "path_to_store_config": path_to_store_config_val,
                "path_to_store_merged_model": path_to_store_merged_model_val,
                "mode": mode_val,
                "metric": metric_val,
            }

            # Only include benchmark if it's not mean mode
            if mode_val != "mean" and bench_val:
                config["bench"] = bench_val

            # Extract multi-language inputs from args
            lang_vals = args[:5]
            model_vals = args[5:10]
            task_vals = args[10:15]

            use_lm_eval = True
            is_multilingual = merge_type_val == "multi"

            if is_multilingual:
                # Multilingual configuration - only use the number of languages specified
                actual_n_langs = int(n_langs_val)
                # Only take the languages that are being used (based on n_langs)
                langs = [lang_vals[i] for i in range(actual_n_langs)]
                models = {langs[i]: model_vals[i] for i in range(actual_n_langs)}

                config["langs"] = langs
                config["models"] = models

                # LM-Eval harness configuration - only include the languages actually used
                search_tasks = {langs[i]: task_vals[i] for i in range(actual_n_langs)}
                # Create a distinct copy of the tasks for test to avoid YAML anchors
                test_tasks = {k: v for k, v in search_tasks.items()}

                config["tasks"] = {"search": search_tasks, "test": test_tasks}
                # Use the multi-task additional templates folder
                config["additional_templates_folder"] = (
                    additional_templates_folder_multi.value
                )
            else:
                # Single language configuration - use task0 for language ID
                config["langs"] = ["task0"]
                config["models"] = {"task0": model_path_single_val}

                # Single language LM-Eval configuration - fix for YAML anchors
                config["tasks"] = {
                    "search": {"task0": task_name_single_val},
                    # Create a distinct copy to avoid YAML anchors
                    "test": {"task0": task_name_single_val},
                }
                config["additional_templates_folder"] = (
                    additional_templates_folder_single_val
                )

            # Save configuration to file
            config_dir = Path(path_to_store_config_val)
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / f"{run_id_val}_config.yaml"

            # Use safe_dump with default_flow_style=False to avoid aliases/anchors
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            # Return preview and file path - also use safe_dump here
            yaml_str = yaml.dump(config, default_flow_style=False)
            config_file = str(config_file)

            # Return yaml preview, file path, and True to indicate config was generated
            return yaml_str, config_file, True, gr.update(interactive=True)

        # Connect the generate_config function with updated outputs
        generate_config_btn.click(
            fn=generate_config,
            inputs=[
                eval_method,
                merge_type,
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                path_to_store_config,
                path_to_store_merged_model,
                n_langs,
                lang_single,
                model_path_single,
                task_name_single,
                metric_single,
                additional_templates_folder_single,
                bench,
                mode,
                metric,
            ]
            + lang_inputs
            + model_inputs
            + task_inputs,
            outputs=[
                config_output,
                config_file_path,
                config_generated,
                go_to_execution_btn,
            ],
        )

        # Connect all input components to the update_preview function
        for component in (
            [
                merge_type,
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                path_to_store_config,
                path_to_store_merged_model,
                n_langs,
                lang_single,
                model_path_single,
                task_name_single,
                metric_single,
                additional_templates_folder_single,
                bench,
                mode,
                metric,
                additional_templates_folder_multi,
            ]
            + lang_inputs
            + model_inputs
            + task_inputs
        ):
            component.change(
                fn=update_preview,
                inputs=[
                    eval_method,
                    merge_type,
                    run_id,
                    base_model,
                    device,
                    dtype,
                    pop_size,
                    n_iter,
                    n_samples,
                    eval_batch_size,
                    seed,
                    path_to_store_config,
                    path_to_store_merged_model,
                    n_langs,
                    lang_single,
                    model_path_single,
                    task_name_single,
                    metric_single,
                    additional_templates_folder_single,
                    bench,
                    mode,
                    metric,
                ]
                + lang_inputs
                + model_inputs
                + task_inputs,
                outputs=config_output,
            )

        # Function to switch to execution tab
        def switch_to_execution_tab():
            return gr.update(selected="execution_tab")

        # Connect the button to switch tabs
        go_to_execution_btn.click(fn=switch_to_execution_tab, inputs=[], outputs=[tabs])

        # Launch experiment function
        def start_experiment(config_file_path, eval_method_val, merge_type_val):
            if not config_file_path:
                return (
                    "Error: No configuration file",
                    "Please generate a configuration first.",
                )

            # Determine which script to run
            script_subdir = "evolutionary-merging-lm-harness"
            script_name = (
                "end2end_multilingual.py" if merge_type_val == "multi" else "end2end.py"
            )
            script_path = PROJECT_ROOT / "experiments" / script_subdir / script_name

            if not script_path.exists():
                return (
                    f"Error: Script not found at {script_path}",
                    f"Script not found: {script_path}",
                )

            # Extract run_id from config file path
            run_id = Path(config_file_path).stem.split("_config")[0]

            return (
                "Experiment starting...",
                f"Starting experiment with script: {script_path}\nConfiguration: {config_file_path}\nRun ID: {run_id}\n",
            )

        # Connect launch button
        launch_btn.click(
            fn=start_experiment,
            inputs=[config_file_path, eval_method, merge_type],
            outputs=[experiment_status, log_output],
        )

        # Add a separate event handler for the experiment execution that properly handles the generator
        def execute_experiment(config_path, eval_method_val, merge_type_val):
            """Execute the experiment and stream the logs."""
            if not config_path:
                return "No configuration file provided."

            script_path = (
                PROJECT_ROOT
                / "experiments"
                / "evolutionary-merging-lm-harness"
                / (
                    "end2end_multilingual.py"
                    if merge_type_val == "multi"
                    else "end2end.py"
                )
            )

            run_id = Path(config_path).stem.split("_config")[0]

            # Call the run_experiment function but handle it as a generator
            for log_line in run_experiment(script_path, config_path, run_id):
                yield log_line

        # Set up the event handler with proper streaming configuration
        launch_btn.click(
            fn=execute_experiment,
            inputs=[config_file_path, eval_method, merge_type],
            outputs=log_output,
            api_name="execute_experiment",
            show_progress=True,
            queue=True,
        )

        stop_btn.click(
            fn=stop_experiment, inputs=[], outputs=[experiment_status, log_output]
        )

        # Replace the existing load button connection logic - add message to outputs
        config_dropdown.change(
            fn=load_and_apply_configuration,
            inputs=[
                config_dropdown,
                eval_method,
                merge_type,
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                path_to_store_config,
                path_to_store_merged_model,
                n_langs,
                lang_single,
                model_path_single,
                task_name_single,
                metric_single,
                additional_templates_folder_single,
                bench,
                mode,
                metric,
            ]
            + lang_inputs
            + model_inputs
            + task_inputs,
            outputs=[
                eval_method,
                merge_type,
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                path_to_store_config,
                path_to_store_merged_model,
                n_langs,
                lang_single,
                model_path_single,
                task_name_single,
                metric_single,
                additional_templates_folder_single,
                bench,
                mode,
                metric,
            ]
            + lang_inputs
            + model_inputs
            + task_inputs
            + [config_output, config_file_path, config_load_message],
        )

        # Function to update configuration dropdown and clear message
        def refresh_configurations():
            """Refresh the list of configurations and clear any messages."""
            configs = list_configurations()
            return gr.update(choices=configs), ""

        # Connect refresh button with updated function
        refresh_btn.click(
            fn=refresh_configurations,
            inputs=[],
            outputs=[config_dropdown, config_load_message],
        )

        # Function to refresh LM-eval tasks
        def refresh_lm_tasks():
            """Refresh the list of LM-eval tasks."""
            fresh_tasks = get_lm_eval_tasks()
            if mode != "mean":
                fresh_tasks = [
                    task for task in fresh_tasks if task in allowed_benchmarks
                ]
            # Update both the single task dropdown and all multilingual task dropdowns
            updates = [gr.Dropdown.update(choices=fresh_tasks)]
            for _ in range(5):  # For all language task dropdowns
                updates.append(gr.Dropdown.update(choices=fresh_tasks))
            return updates

        # Add a refresh button for LM-eval tasks (hidden in the UI unless needed)
        refresh_tasks_btn = gr.Button("Refresh Task List", visible=False)
        refresh_tasks_btn.click(
            fn=refresh_lm_tasks, inputs=[], outputs=[task_name_single] + task_inputs
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", share=True)
