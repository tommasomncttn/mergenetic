# import required modules
import argparse
import logging
import os
import sys

# Set up logging, instead of sending logs to stderr, use stdout
# and set the format to include the timestamp, level, and message
from logging import getLogger

import numpy as np
import yaml
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from mergenetic import PROJECT_ROOT
from mergenetic.evaluation.utils import evaluate_models_lm_harness, retrieve_thetas
from mergenetic.merging import SlerpMerger
from mergenetic.optimization.predefined_problems import (
    ConfigPE,
    CrossLingualMathProblem,
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

    lang_id = config.langs[0]
    path = f"{PROJECT_ROOT}/{config.additional_templates_folder}"
    task_manager = TaskManager(include_path=path)
    task_name = config.tasks["search"][lang_id]
    task: ConfigurableTask = task_manager.load_task_or_group(task_name)[task_name]

    anchors = np.random.choice(
        range(len(task.dataset["test"])), config.n_samples, replace=False
    )
    anchors_weights = np.ones(len(anchors)) / len(anchors)
    anchors, anchors_weights

    logger.info(f"STEP 1 completed: Anchors extracted: {anchors}")

    if config.mode != "mean":
        thetas_paths = {
            k: f"{config.path_to_store_config}/{m.split('/')[-1]}_{k}_theta.pkl"
            for k, m in config.models.items()
        }

        if all(os.path.exists(thetas_paths[l]) for l in config.langs):
            logger.info(f"Thetas for {lang_id} already exist. Loading them.")
            thetas = retrieve_thetas(config)
        else:
            # STEP 3. Get the responses of the base models
            _ = evaluate_models_lm_harness(config)
            logger.info("STEP 2 completed: Predictions of base models obtained")

        # STEP 3. Get the thetas
        thetas = retrieve_thetas(config)
        logger.info("STEP 3 completed: Thetas obtained")
    else:
        thetas = {}

    # STEP 4. Unpack some parameters and set the accuracy estimation parameters
    pop_size = config.pop_size
    n_iter = config.n_iter
    run_id = config.run_id
    bench = config.bench
    mode = config.mode

    thetas_list = list(thetas.values()) if thetas else []
    est_parameters = ConfigPE(
        thetas=thetas_list,
        weights=anchors_weights,
        sample_ids=anchors,
        bench=bench,
        mode=mode,
        correct_metric=config.metric,
    )

    logger.info("STEP 4 completed: Parameters unpacked and set: %s", est_parameters)

    # STEP 7. Define the merger
    path_to_store_yaml = f"{config.path_to_store_config}/{config.run_id}/"
    merger = SlerpMerger(
        run_id=run_id,
        path_to_base_model=config.base_model,
        path_to_model_1=config.models[lang_id],
        path_to_store_yaml=path_to_store_yaml,
        path_to_store_merged_model=config.path_to_store_merged_model,
        dtype=config.dtype,
        layer_range_base_model=[0, 32],
        layer_range_model_1=[0, 32],
    )

    if config.device:
        device = config.device
    else:
        device = "cuda"
    logger.info("STEP 7 completed: Merger defined")

    # STEP 8. Define the problem
    problem = CrossLingualMathProblem(
        merger,
        test_df=None,
        search_df=None,
        lm_eval_tasks=config.tasks,
        lang_id=lang_id,
        conf_pe=est_parameters,
        device=device,
        n_var=11,
        n_obj=1,
        n_eq_constr=0,
        n_ieq_constr=0,
        discrete=True,
        eager_mode=config.eager_mode,
        load_in_4bit=config.load_in_4bit,
        eval_batch_size=config.eval_batch_size,
        additional_templates_folder=config.additional_templates_folder,
    )
    logger.info("STEP 8 completed: Problem defined")

    # STEP 9. Define the algorithm
    algorithm = GA(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(),
        mutation=PM(),
        eliminate_duplicates=True,
    )
    logger.info("STEP 9 completed: Algorithm defined")

    # STEP 10. Define the searcher and run it
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
    searcher.search()
    logger.info("STEP 10 completed: Searcher defined and run")

    searcher.test()

    logger.info("Search completed. Results saved to %s", results_path)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Evolving the merging of some models.")
    parser.add_argument(
        "--config", type=str, help="The path to the configuration file."
    )
    parser.add_argument(
        "--seed", type=int, help="The seed to use for the experiment.", default=None
    )
    parser.add_argument(
        "--device", type=str, help="The device to use for the experiment.", default=None
    )
    parser.add_argument("--run_id", type=str, help="The id of the run.", default=None)
    parser.add_argument(
        "--n_samples",
        type=int,
        help="The number of samples to use for the experiment.",
        default=None,
    )
    args = parser.parse_args()

    # load the configuration
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config = ConfigLmEval(**config)
    if args.seed:
        config.seed = args.seed
        logger.info(f"Overwriting seed with arg: {args.seed}")

    if args.device:
        config.device = args.device
        logger.info(f"Overwriting device with arg: {args.device}")

    if args.run_id:
        config.run_id = args.run_id
        logger.info(f"Overwriting run_id with arg: {args.run_id}")

    if args.n_samples:
        config.n_samples = args.n_samples
        logger.info(f"Overwriting n_samples with arg: {args.n_samples}")

    logger.info(f"Starting the experiment with the following configuration: {config}")

    main(config)
