import logging
from pathlib import Path
import yaml
import gradio as gr

from mergenetic_gui.gui_utils import (
    PROJECT_ROOT,
    list_configurations,
    load_configuration,
    run_experiment,
)

# ---------------------------
# Constants & logging
# ---------------------------
ALLOWED_BENCHMARKS = [
    "arc",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "truthfulqa",
    "winogrande",
]
EVAL_MODES = ["mean", "irt", "pirt", "mpirt", "gmpirt"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mergenetic.gui.single")


# ---------------------------
# Core config helpers (single-task)
# ---------------------------

def _to_int(x, default: int):
    try:
        return int(float(x))
    except Exception:
        return default


def build_config_dict_single(
    run_id_val,
    base_model_val,
    device_val,
    dtype_val,
    pop_size_val,
    n_iter_val,
    n_samples_val,
    eval_batch_size_val,
    seed_val,
    cfg_dir_val,
    model_dir_val,
    mode_val,
    bench_val,
    metric_val,
    model_path_single,
    task_single,
    add_tpls_single,
):
    """Create the python dict for the YAML config from current UI values (single task)."""
    pop_size_val = _to_int(pop_size_val, 10)
    n_iter_val = _to_int(n_iter_val, 10)
    n_samples_val = _to_int(n_samples_val, 10)
    eval_batch_size_val = _to_int(eval_batch_size_val, 32)
    seed_val = _to_int(seed_val, 42)

    config = {
        "run_id": run_id_val,
        "base_model": base_model_val,
        "device": device_val,
        "dtype": dtype_val,
        "task_type": "lm_eval",
        "pop_size": pop_size_val,
        "n_iter": n_iter_val,
        "n_samples": n_samples_val,
        "eval_batch_size": eval_batch_size_val,
        "seed": seed_val,
        "path_to_store_config": cfg_dir_val,
        "path_to_store_merged_model": model_dir_val,
        "mode": mode_val,
        "metric": metric_val,
        "langs": ["task0"],
        "models": {"task0": model_path_single},
        "tasks": {
            "search": {"task0": task_single},
            "test": {"task0": task_single},
        },
        "additional_templates_folder": add_tpls_single,
    }

    if mode_val != "mean" and bench_val:
        config["bench"] = bench_val

    return config


def yaml_preview_from_ui_single(*args):
    cfg = build_config_dict_single(*args)
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def generate_config_and_save_single(*args):
    cfg = build_config_dict_single(*args)
    cfg_dir = Path(cfg["path_to_store_config"])  # type: ignore
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_file = cfg_dir / f"{cfg['run_id']}_config.yaml"  # type: ignore
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return (
        yaml.dump(cfg, default_flow_style=False, sort_keys=False),
        str(config_file),
        True,
    )


# ---------------------------
# Small UI helpers
# ---------------------------

def update_benchmark_visibility(mode_val):
    if mode_val == "mean":
        return gr.update(interactive=False, value="", choices=[""])
    else:
        return gr.update(interactive=True, value="gsm8k", choices=ALLOWED_BENCHMARKS)


def refresh_configurations():
    return list_configurations()


# ---------------------------
# Loading existing config into UI
# ---------------------------

def load_config_to_ui_single(
    selection,
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
    model_path_single,
    task_single,
    add_tpls_single,
):
    data = load_configuration(selection)
    if not data:
        # New config — keep current values, refresh run_id note via message
        msg = "<span style='color:#33691e;font-weight:bold;'>Started new configuration.</span>"
        return (
            gr.update(value=run_id),
            gr.update(value=base_model),
            gr.update(value=device),
            gr.update(value=dtype),
            gr.update(value=pop_size),
            gr.update(value=n_iter),
            gr.update(value=n_samples),
            gr.update(value=eval_batch_size),
            gr.update(value=seed),
            gr.update(value=cfg_dir),
            gr.update(value=model_dir),
            gr.update(value=mode),
            gr.update(value=bench),
            gr.update(value=metric),
            gr.update(value=model_path_single),
            gr.update(value=task_single),
            gr.update(value=add_tpls_single),
            yaml.dump({}, default_flow_style=False, sort_keys=False),
            "",
            msg,
        )

    config, config_file = data
    logger.info("LOADED CONFIG (single): %s", config)

    # Map fields
    run_id_val = config.get("run_id", run_id)
    base_model_val = config.get("base_model", base_model)
    device_val = config.get("device", device)
    dtype_val = config.get("dtype", dtype)
    pop_size_val = config.get("pop_size", pop_size)
    n_iter_val = config.get("n_iter", n_iter)
    n_samples_val = config.get("n_samples", n_samples)
    eval_batch_size_val = config.get("eval_batch_size", eval_batch_size)
    seed_val = config.get("seed", seed)
    cfg_dir_val = config.get("path_to_store_config", cfg_dir)
    model_dir_val = config.get("path_to_store_merged_model", model_dir)
    mode_val = config.get("mode", mode)
    metric_val = config.get("metric", metric)

    # bench is irrelevant for mean
    bench_val = config.get("bench", "") if mode_val == "mean" else config.get("bench", bench)

    models = config.get("models", {})
    tasks = config.get("tasks", {}).get("search", {})

    model_path_single_val = models.get("task0", model_path_single)
    task_single_val = tasks.get("task0", task_single)
    add_tpls_single_val = config.get("additional_templates_folder", add_tpls_single)

    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    msg = "<span style='color:#2e7d32;font-weight:bold;'>✅ Configuration Loaded</span>"

    return (
        gr.update(value=run_id_val),
        gr.update(value=base_model_val),
        gr.update(value=device_val),
        gr.update(value=dtype_val),
        gr.update(value=pop_size_val),
        gr.update(value=n_iter_val),
        gr.update(value=n_samples_val),
        gr.update(value=eval_batch_size_val),
        gr.update(value=seed_val),
        gr.update(value=cfg_dir_val),
        gr.update(value=model_dir_val),
        gr.update(value=mode_val),
        gr.update(value=bench_val),
        gr.update(value=metric_val),
        gr.update(value=model_path_single_val),
        gr.update(value=task_single_val),
        gr.update(value=add_tpls_single_val),
        yaml_str,
        config_file,
        msg,
    )


# ---------------------------
# Execution helpers (single)
# ---------------------------

def start_experiment_single(config_file_path):
    if not config_file_path:
        return ("Error: No configuration file", "Please generate a configuration first.")

    script_path = PROJECT_ROOT / "experiments" / "evolutionary-merging-lm-harness" / "end2end.py"
    if not script_path.exists():
        return (
            f"Error: Script not found at {script_path}",
            f"Script not found: {script_path}",
        )

    run_id = Path(config_file_path).stem.split("_config")[0]
    return (
        "Experiment starting...",
        f"Starting experiment with script: {script_path}\nConfiguration: {config_file_path}\nRun ID: {run_id}\n",
    )


def execute_experiment_single(config_path):
    if not config_path:
        yield "No configuration file provided."
        return

    script_path = PROJECT_ROOT / "experiments" / "evolutionary-merging-lm-harness" / "end2end.py"
    run_id = Path(config_path).stem.split("_config")[0]

    for log_line in run_experiment(script_path, config_path, run_id):
        yield log_line