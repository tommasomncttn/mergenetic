# import required modules
import argparse
import logging
import sys

# Set up logging, instead of sending logs to stderr, use stdout
# and set the format to include the timestamp, level, and message
from logging import getLogger

import numpy as np
import torch
import yaml
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from pymoo.algorithms.moo.nsga2 import NSGA2

from mergenetic import PROJECT_ROOT
from mergenetic.evaluation.utils import evaluate_models_lm_harness, retrieve_thetas
from mergenetic.merging import TiesDareMerger
from mergenetic.optimization.predefined_problems import (
    ConfigMultiLingualPE,
    MultilingualMergingProblem,
)
from mergenetic.searcher import Searcher
from mergenetic.utils import ConfigLmEval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = getLogger(__name__)


def main(config: ConfigLmEval):

    if config.mode != "mean":
        evaluate_models_lm_harness(config)
        logger.info("STEP 1 completed: Predictions of base models obtained")

        # STEP 2. Get the thetas
        thetas = retrieve_thetas(config)
        logger.info("STEP 2 completed: Thetas obtained")
    else:
        thetas = {}
        for lang in config.langs:
            thetas[lang] = []

    # STEP 3. Extract the anchor points
    anchors = {}
    anchors_weights = {}
    for lang in config.langs:
        path = f"{PROJECT_ROOT}/{config.additional_templates_folder}"
        task_manager = TaskManager(include_path=path)
        task_name = config.tasks["search"][lang]
        task: ConfigurableTask = task_manager.load_task_or_group(task_name)[task_name]

        anchors[lang] = np.random.choice(
            range(len(task.dataset["test"])), config.n_samples, replace=False
        )
        anchors_weights[lang] = np.ones(len(anchors[lang])) / len(anchors[lang])

    logger.info("STEP 3 completed: Anchors extracted; anchors: %s", anchors)

    # STEP 4. Unpack some parameters and set the accuracy estimation parameters
    pop_size = config.pop_size
    n_iter = config.n_iter
    run_id = config.run_id
    bench = config.bench
    mode = config.mode
    tasks = config.tasks
    metric = config.metric

    thetas = [thetas[lang] for lang in config.langs]
    conf_pe = ConfigMultiLingualPE(
        sample_ids=anchors,
        weights=anchors_weights,
        correct_metric=metric,
        thetas=thetas,
        bench=bench,
        mode=mode,
    )

    # STEP 5. Define the merger
    base_model = config.base_model
    model_paths = []
    for key in config.langs:
        model_paths.append(config.models[key])

    path_to_store_yaml = f"{config.path_to_store_config}/{config.run_id}/"
    merger = TiesDareMerger(
        run_id=run_id,
        path_to_base_model=base_model,
        model_paths=model_paths,
        path_to_store_yaml=path_to_store_yaml,
        path_to_store_merged_model=config.path_to_store_merged_model,
        dtype=config.dtype,
    )

    if config.device:
        device = config.device
    else:
        device = "cuda"

    # STEP 6. Define the problem

    n_var = len(model_paths) * 2
    problem = MultilingualMergingProblem(
        merger,
        lm_eval_tasks=tasks,
        search_df_dict=None,
        test_df_dict=None,
        config_pe=conf_pe,
        n_var=n_var,
        n_obj=len(tasks["search"])
        - 1,  # -1 because the last task is only used to evaluate the base model
        n_eq_constr=0,
        n_ieq_constr=0,
        eval_batch_size=config.eval_batch_size,
        device=device,
        detect_lang=False,
        eager_mode=config.eager_mode,
        load_in_4bit=config.load_in_4bit,
        additional_templates_folder=config.additional_templates_folder,
    )

    # STEP 7. Define the algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True,
    )

    # STEP 8. Define the searcher and run it
    results_path = f"{config.path_to_store_config}/{config.run_id}/"
    searcher = Searcher(
        problem,
        algorithm,
        results_path,
        n_iter,
        run_id=run_id,
        seed=config.seed,
        verbose=False,
    )

    logger.info("Starting the search...")

    searcher.search()
    searcher.test()

    logger.info("Search completed. Results saved to %s", results_path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Evolving the merging of some models.")
    parser.add_argument(
        "--config", type=str, help="The path to the configuration file."
    )
    parser.add_argument(
        "--device", type=str, help="The device to use for the experiment.", default=None
    )
    parser.add_argument("--run_id", type=str, help="The id of the run.", default=None)
    parser.add_argument(
        "--no-preeval",
        action="store_true",
        help="Whether to skip the pre-evaluation step.",
    )

    args = parser.parse_args()

    # load the configuration
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config = ConfigLmEval(**config)
    logger.info(f"Configuration loaded: {config}")

    if args.device:
        config.device = args.device
        logger.info(f"Overwriting device with arg: {args.device}")

    if args.run_id:
        config.run_id = args.run_id
        logger.info(f"Overwriting run_id with arg: {args.run_id}")

    logger.info(f"Starting the experiment with the following configuration: {config}")

    main(config)
