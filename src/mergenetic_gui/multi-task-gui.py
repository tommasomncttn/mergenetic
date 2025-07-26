import uuid
import gradio as gr

from mergenetic_gui.gui_utils import (
    PROJECT_ROOT,
    create_config_directory,
    get_lm_eval_tasks,
    stop_experiment,
)

# Local callbacks for multi-task UI
from callbacks_multi import (
    EVAL_MODES,
    ALLOWED_BENCHMARKS,
    MAX_TASKS,
    yaml_preview_from_ui_multi,
    generate_config_and_save_multi,
    update_benchmark_visibility,
    update_task_visibility,
    start_experiment_multi,
    execute_experiment_multi,
    refresh_configurations,
    load_config_to_ui_multi,
)


def create_interface():
    lm_eval_tasks = get_lm_eval_tasks()

    with gr.Blocks(title="Mergenetic GUI (Multi-Task – Single Page)") as demo:
        gr.Markdown("# Mergenetic: Evolutionary Model Merging — Multi‑Task UI")
        gr.Markdown(
            "Single-page multi-task UI. Keep first-order fields visible; other knobs in one accordion."
        )

        # --- Top bar: configuration selector --------------------------------------------------
        with gr.Row(equal_height=True):
            config_dropdown = gr.Dropdown(
                choices=refresh_configurations(),
                label="Load configuration",
                value="-- New Configuration --",
            )
            refresh_cfg_btn = gr.Button("🔄", variant="secondary", min_width=60)

        config_load_msg = gr.HTML(visible=True)

        # --- Essentials ----------------------------------------------------------------------
        run_id = gr.Textbox(label="Run ID", value=f"run_{uuid.uuid4().hex[:8]}")
        base_model = gr.Textbox(
            label="Base model path",
            value="mistralai/Mistral-7B-v0.1",
        )

        with gr.Row():
            mode = gr.Dropdown(
                choices=EVAL_MODES, value="gmpirt", label="Evaluation mode"
            )
            bench = gr.Dropdown(
                choices=ALLOWED_BENCHMARKS,
                value="gsm8k",
                label="Benchmark (disabled when mean)",
            )

        # --- Multi-task fields ---------------------------------------------------------------
        gr.Markdown("### Tasks")
        n_tasks = gr.Number(
            label="#Tasks",
            value=2,
            minimum=1,
            maximum=MAX_TASKS,
            precision=0,
        )

        # Per-task rows (ID, model path, LM-Eval task)
        id_inputs, model_inputs, task_inputs, row_groups = [], [], [], []
        for i in range(MAX_TASKS):
            with gr.Row(visible=i < 2) as row:
                id_box = gr.Textbox(label=f"Task {i} ID", value=f"task{i}")
                model_box = gr.Textbox(label=f"Model path {i}", value=f"model_task{i}")
                task_box = gr.Dropdown(
                    choices=lm_eval_tasks,
                    allow_custom_value=True,
                    value=(
                        "gsm8k"
                        if lm_eval_tasks and "gsm8k" in lm_eval_tasks
                        else (lm_eval_tasks[0] if lm_eval_tasks else "")
                    ),
                    label=f"LM-Eval task {i}",
                )
            id_inputs.append(id_box)
            model_inputs.append(model_box)
            task_inputs.append(task_box)
            row_groups.append(row)

        add_tpls_multi = gr.Textbox(
            label="Additional tasks folder", value="lm_tasks"
        )

        # Toggle row visibility when #tasks changes
        n_tasks.change(
            update_task_visibility,
            inputs=n_tasks,
            outputs=row_groups,   # we hide/show the entire rows
        )

        # --- Optional: Advanced optimisation & system -----------------------------------------
        with gr.Accordion("Advanced optimisation & system (optional)", open=False):
            pop_size = gr.Number(label="Population size", value=10, precision=0)
            n_iter = gr.Number(label="#Iterations", value=10, precision=0)
            n_samples = gr.Number(label="#Samples", value=10, precision=0)
            eval_batch_size = gr.Number(label="Eval batch size", value=32, precision=0)
            seed = gr.Number(label="Random seed", value=42, precision=0)

            device = gr.Textbox(label="Device", value="cuda")
            dtype = gr.Dropdown(
                ["float16", "float32", "bfloat16"], value="float16", label="dtype"
            )
            metric = gr.Textbox(label="Metric (lm-eval)", value="exact_match")

            cfg_dir = gr.Textbox(
                label="Path to store config", value=str(create_config_directory())
            )
            model_dir = gr.Textbox(
                label="Path to store merged model", value=str(PROJECT_ROOT / "models")
            )

        # Mean disables benchmark *without* changing choices/value (prevents validation error)
        mode.change(update_benchmark_visibility, inputs=mode, outputs=bench)

        # --- Config preview + actions ---------------------------------------------------------
        cfg_preview = gr.Code(label="Configuration preview (YAML)", language="yaml")
        cfg_path = gr.Textbox(label="Configuration file path", interactive=False)
        cfg_generated = gr.State(value=False)

        with gr.Row():
            gen_btn = gr.Button("Generate configuration", variant="primary")
            launch_btn = gr.Button("Launch experiment", variant="secondary")
            stop_btn = gr.Button("Stop experiment", variant="stop")

        experiment_status = gr.Textbox(
            label="Experiment status", value="Not started", interactive=False
        )
        log_output = gr.TextArea(
            label="Experiment log", interactive=False, lines=20, autoscroll=True
        )

        # --- Wire callbacks -------------------------------------------------------------------
        preview_inputs = [
            run_id,
            base_model,
            device,
            dtype,
            pop_size,
            n_iter,
            n_samples,
            eval_batch_size,
            seed,
            cfg_dir,
            model_dir,
            mode,
            bench,
            metric,
            n_tasks,
        ] + id_inputs + model_inputs + task_inputs + [
            add_tpls_multi,
        ]

        # Auto-update YAML preview on change
        for c in preview_inputs:
            c.change(
                yaml_preview_from_ui_multi,
                inputs=preview_inputs,
                outputs=cfg_preview,
            )

        # Generate config button
        gen_btn.click(
            fn=generate_config_and_save_multi,
            inputs=preview_inputs,
            outputs=[cfg_preview, cfg_path, cfg_generated],
        )

        # Execution buttons
        launch_btn.click(
            fn=start_experiment_multi,
            inputs=[cfg_path],
            outputs=[experiment_status, log_output],
        )
        launch_btn.click(
            fn=execute_experiment_multi,
            inputs=[cfg_path],
            outputs=log_output,
            show_progress=True,
            queue=True,
        )
        stop_btn.click(
            fn=stop_experiment, inputs=[], outputs=[experiment_status, log_output]
        )

        # Refresh config list
        refresh_cfg_btn.click(
            refresh_configurations, inputs=[], outputs=config_dropdown
        )

        # Load config -> fill UI
        config_dropdown.change(
            fn=load_config_to_ui_multi,
            inputs=[
                config_dropdown,
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                cfg_dir,
                model_dir,
                mode,
                bench,
                metric,
                n_tasks,
            ]
            + id_inputs
            + model_inputs
            + task_inputs
            + [add_tpls_multi],
            outputs=[
                run_id,
                base_model,
                device,
                dtype,
                pop_size,
                n_iter,
                n_samples,
                eval_batch_size,
                seed,
                cfg_dir,
                model_dir,
                mode,
                bench,
                metric,
                n_tasks,
            ]
            + id_inputs
            + model_inputs
            + task_inputs
            + [
                add_tpls_multi,
                cfg_preview,
                cfg_path,
                config_load_msg,
            ],
        )

    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.queue().launch(server_name="0.0.0.0", share=True)
