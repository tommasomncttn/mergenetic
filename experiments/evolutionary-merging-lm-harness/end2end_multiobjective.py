# import required modules
import argparse
import logging
import sys

# Set up logging, instead of sending logs to stderr, use stdout
# and set the format to include the timestamp, level, file:line, and message
from logging import getLogger

import numpy as np
import yaml
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from pymoo.algorithms.moo.nsga2 import NSGA2

from mergenetic import PROJECT_ROOT
from mergenetic.merging import TiesDareMerger
from mergenetic.optimization.predefined_problems import (
    ConfigLmEvalMultiObjectivePE,
    LmEvalMultiObjectiveProblem,
)
from mergenetic.searcher import Searcher
from mergenetic.utils import ConfigMultiObjective

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = getLogger(__name__)


def main(config: ConfigMultiObjective):

    # STEP 3. Extract the anchor points
    anchors = {}
    anchors_weights = {}
    for task in config.tasks:
        path = f"{PROJECT_ROOT}/{config.additional_templates_folder}"
        task_manager = TaskManager(include_path=path)
        ctask: ConfigurableTask = task_manager.load_task_or_group(task)[task]

        anchors[task] = np.random.choice(
            range(len(ctask.dataset["test"])), config.n_samples, replace=False
        )
        anchors_weights[task] = np.ones(len(anchors[task])) / len(anchors[task])

    logger.info("STEP 3 completed: Anchors extracted; anchors: %s", anchors)

    # STEP 4. Unpack some parameters and set the accuracy estimation parameters
    pop_size = config.pop_size
    n_iter = config.n_iter
    run_id = config.run_id
    tasks = config.tasks
    metric = config.metric

    print(f"Passing anchors: {anchors}")

    conf_pe = ConfigLmEvalMultiObjectivePE(
        tasks=tasks,
        sample_ids=anchors,
        correct_metric=metric,
        additional_templates_folder=config.additional_templates_folder,
    )

    # STEP 5. Define the merger
    base_model = config.base_model
    model_paths = config.models

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
    problem = LmEvalMultiObjectiveProblem(
        config=conf_pe,
        merger=merger,
        n_var=n_var,
        n_obj=len(tasks),
        n_eq_constr=0,
        n_ieq_constr=0,
        device=device,
        load_in_4bit=config.load_in_4bit,
        eval_batch_size=config.eval_batch_size,
        eager_mode=config.eager_mode,
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

    config = ConfigMultiObjective(**config)
    logger.info(f"Configuration loaded: {config}")

    if args.device:
        config.device = args.device
        logger.info(f"Overwriting device with arg: {args.device}")

    if args.run_id:
        config.run_id = args.run_id
        logger.info(f"Overwriting run_id with arg: {args.run_id}")

    if args.device or args.run_id:
        logger.info(
            f"Overwrote configuration. Starting the experiment with the following configuration: {config}"
        )
    else:
        logger.info("No changes in the configuration.")

    main(config)
