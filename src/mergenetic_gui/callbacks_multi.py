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
MAX_TASKS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mergenetic.gui.multi")


# ---------------------------
# Core config helpers (multi-task)
# ---------------------------
def _to_int(x, default: int):
    try:
        return int(float(x))
    except Exception:
        return default


def build_config_dict_multi(
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
    n_tasks_val,
    *lists,  # ids[0..k-1], model_paths[0..k-1], tasks[0..k-1], add_tpls_multi (last)
):
    """
    Build YAML-ready config for multi-task.
    lists: ids + model_paths + tasks + [add_tpls_multi]
    """
    pop_size_val = _to_int(pop_size_val, 10)
    n_iter_val = _to_int(n_iter_val, 10)
    n_samples_val = _to_int(n_samples_val, 10)
    eval_batch_size_val = _to_int(eval_batch_size_val, 32)
    seed_val = _to_int(seed_val, 42)
    n = _to_int(n_tasks_val, 2)
    n = max(1, min(MAX_TASKS, n))

    # Split the flattened lists
    ids = lists[:MAX_TASKS]
    models = lists[MAX_TASKS:2 * MAX_TASKS]
    tasks = lists[2 * MAX_TASKS:3 * MAX_TASKS]
    add_tpls_multi = lists[3 * MAX_TASKS] if len(lists) > 3 * MAX_TASKS else "lm_tasks"

    # Only keep the first n items
    lang_ids = [(ids[i] if ids[i] else f"task{i}") for i in range(n)]
    model_paths = [models[i] for i in range(n)]
    task_names = [tasks[i] for i in range(n)]

    cfg = {
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
        "langs": lang_ids,
        "models": {lang_ids[i]: model_paths[i] for i in range(n)},
        "tasks": {
            "search": {lang_ids[i]: task_names[i] for i in range(n)},
            "test":   {lang_ids[i]: task_names[i] for i in range(n)},  # separate copy
        },
        "additional_templates_folder": add_tpls_multi,
    }

    if mode_val != "mean" and bench_val:
        cfg["bench"] = bench_val

    return cfg


def yaml_preview_from_ui_multi(*args):
    cfg = build_config_dict_multi(*args)
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def generate_config_and_save_multi(*args):
    cfg = build_config_dict_multi(*args)
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
    """
    Disable the benchmark dropdown on 'mean' without changing its value/choices.
    This avoids the 'value not in choices' Gradio error.
    """
    if mode_val == "mean":
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True, value="gsm8k", choices=ALLOWED_BENCHMARKS)


def update_task_visibility(n_tasks):
    try:
        n = int(float(n_tasks))
    except Exception:
        n = 2
    n = max(1, min(MAX_TASKS, n))
    # Return visibility updates for each row group
    return [gr.update(visible=i < n) for i in range(MAX_TASKS)]


def refresh_configurations():
    return list_configurations()


# ---------------------------
# Loading existing config into UI (multi)
# ---------------------------
def load_config_to_ui_multi(
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
    n_tasks,
    *lists,  # ids, models, tasks, add_tpls_multi
):
    data = load_configuration(selection)
    # lists = [ids0..ids4, models0..models4, tasks0..tasks4, add_tpls_multi]
    ids = lists[:MAX_TASKS]
    models = lists[MAX_TASKS:2 * MAX_TASKS]
    tasks = lists[2 * MAX_TASKS:3 * MAX_TASKS]
    add_tpls_multi = lists[3 * MAX_TASKS]

    if not data:
        # New config — keep current values; show message
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
            gr.update(value=n_tasks),
            *[gr.update(value=ids[i]) for i in range(MAX_TASKS)],
            *[gr.update(value=models[i]) for i in range(MAX_TASKS)],
            *[gr.update(value=tasks[i]) for i in range(MAX_TASKS)],
            gr.update(value=add_tpls_multi),
            yaml.dump({}, default_flow_style=False, sort_keys=False),
            "",
            msg,
        )

    config, config_file = data
    logger.info("LOADED CONFIG (multi): %s", config)

    # Map base/global fields
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
    bench_val = config.get("bench", bench)  # keep valid value even if mean

    tasks_dict = config.get("tasks", {}).get("search", {})
    langs = config.get("langs", [])
    models_dict = config.get("models", {})
    add_tpls_multi_val = config.get("additional_templates_folder", add_tpls_multi)

    # Fill arrays
    num = min(MAX_TASKS, max(1, len(langs) if langs else len(tasks_dict) or 1))
    ids_vals = [ids[i] for i in range(MAX_TASKS)]
    model_vals = [models[i] for i in range(MAX_TASKS)]
    task_vals = [tasks[i] for i in range(MAX_TASKS)]

    # Prefer langs; fall back to task keys
    if langs:
        keys = langs[:num]
    else:
        keys = list(tasks_dict.keys())[:num] or [f"task{i}" for i in range(num)]

    for i, k in enumerate(keys):
        ids_vals[i] = k
        model_vals[i] = models_dict.get(k, model_vals[i])
        task_vals[i] = tasks_dict.get(k, task_vals[i])

    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    msg = "<span style='color:#2e7d32;font-weight:bold;'>✅ Configuration Loaded</span>"

    # Build output tuple
    out = [
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
        gr.update(value=num),
    ]
    out += [gr.update(value=v) for v in ids_vals]
    out += [gr.update(value=v) for v in model_vals]
    out += [gr.update(value=v) for v in task_vals]
    out += [
        gr.update(value=add_tpls_multi_val),
        yaml_str,
        config_file,
        msg,
    ]
    return tuple(out)


# ---------------------------
# Execution helpers (multi)
# ---------------------------
def start_experiment_multi(config_file_path):
    if not config_file_path:
        return ("Error: No configuration file", "Please generate a configuration first.")

    script_path = PROJECT_ROOT / "experiments" / "evolutionary-merging-lm-harness" / "end2end_multilingual.py"
    if not script_path.exists():
        return (f"Error: Script not found at {script_path}", f"Script not found: {script_path}")

    run_id = Path(config_file_path).stem.split("_config")[0]
    return (
        "Experiment starting...",
        f"Starting experiment with script: {script_path}\nConfiguration: {config_file_path}\nRun ID: {run_id}\n",
    )


def execute_experiment_multi(config_path):
    if not config_path:
        yield "No configuration file provided."
        return

    script_path = PROJECT_ROOT / "experiments" / "evolutionary-merging-lm-harness" / "end2end_multilingual.py"
    run_id = Path(config_path).stem.split("_config")[0]

    for log_line in run_experiment(script_path, config_path, run_id):
        yield log_line
