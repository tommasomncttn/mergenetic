# import required modules
import argparse
import warnings

import pandas as pd
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2

from mergenetic.evaluation.utils import evaluate_models, retrieve_thetas
from mergenetic.merging import TiesDareMerger
from mergenetic.optimization.predefined_problems import (
    ConfigMultiLingualPE,
    MultilingualMergingProblem,
)
from mergenetic.searcher import Searcher
from mergenetic.utils import Config

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
from logging import getLogger

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


def main(config: Config, no_preeval: bool = False):
    # STEP 1. Load the data
    datasets: dict[str, pd.DataFrame] = {}
    for lang in config.langs:
        datasets[lang] = pd.read_csv(config.datasets[lang])

    logger.info(f"STEP 1 completed: Data loaded: {datasets.keys()}")

    # STEP 2. Extract the anchor points
    anchors = {
        key: datasets[key].sample(random_state=config.seed, n=config.n_samples).index
        for key in datasets.keys()
    }
    anchors_weights = {
        key: [1 / config.n_samples] * config.n_samples for key in datasets.keys()
    }
    logger.info(f"STEP 2 completed: Anchors extracted: {anchors}")

    # STEP 3. Get the responses of the base models
    if no_preeval:
        logger.info("Skipping the pre-evaluation step")
    else:
        evaluate_models(config)
        logger.info("STEP 3 completed: Predictions of base models obtained")

    # STEP 4. Get the thetas
    thetas = retrieve_thetas(config)
    logger.info(f"STEP 4 completed: Thetas obtained: {thetas}")

    # STEP 5. Unpack some parameters and set the accuracy estimation parameters
    pop_size = config.pop_size
    n_iter = config.n_iter
    run_id = config.run_id
    bench = config.bench
    mode = config.mode
    thetas_list = list(thetas.values())

    est_params = ConfigMultiLingualPE(
        sample_ids=anchors,
        weights=anchors_weights,
        thetas=thetas_list,
        bench=bench,
        mode=mode,
    )

    logger.info("STEP 5 completed: Parameters unpacked")

    if config.device:
        device = config.device
    else:
        device = "cuda"

    # STEP 6. Extract the samples using the anchors as index and the test set without those samples
    sampled_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for key, df in datasets.items():
        ids = anchors[key]
        sampled_dfs[key] = df.loc[ids].copy()
        test_dfs[key] = df.drop(ids)

        print(sampled_dfs[key], flush=True)
    logger.info(f"STEP 6 completed: Samples extracted: {sampled_dfs}")

    # STEP 7. Define the merger
    base_model = config.base_model
    model_paths = []
    for key in config.models.keys():
        model_paths.append(config.models[key])

    # 3. define the merger
    path_to_store_yaml = f"{config.path_to_store_config}/{run_id}"
    merger = TiesDareMerger(
        run_id=run_id,
        path_to_base_model=base_model,
        model_paths=model_paths,
        path_to_store_yaml=path_to_store_yaml,
        path_to_store_merged_model=config.path_to_store_merged_model,
        dtype=config.dtype,
    )

    logger.info("STEP 7 completed: Merger defined")

    n_var = len(list(config.models.keys())) * 2

    # STEP 8. Define the problem
    problem = MultilingualMergingProblem(
        merger,
        search_df_dict=sampled_dfs,
        test_df_dict=test_dfs,
        n_var=n_var,
        config_pe=est_params,
        n_obj=len(sampled_dfs.keys()),
        n_eq_constr=0,
        n_ieq_constr=0,
        eval_batch_size=32,
        device=device,
        load_in_4bit=True,
    )

    logger.info("STEP 8 completed: Problem defined")

    # STEP 9. Define the algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True,
    )
    logger.info("STEP 9 completed: Algorithm defined")

    # STEP 10. Define the searcher and run it
    result_path = "experiments/evolutionary-merging"
    searcher = Searcher(
        problem,
        algorithm,
        result_path,
        n_iter,
        run_id=run_id,
        seed=config.seed,
        verbose=False,
    )
    searcher.search()

    logger.info("STEP 10 completed: Searcher finished")

    searcher.test()

    logger.info("Search completed. Results saved to %s", result_path)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Evolving the merging of some models.")
    parser.add_argument(
        "--config", type=str, help="The path to the configuration file."
    )
    parser.add_argument("--run_id", type=str, help="The id of the run.")
    parser.add_argument(
        "--no-preeval",
        action="store_true",
        help="Whether to skip the pre-evaluation step.",
    )
    args = parser.parse_args()

    # load the configuration
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config = Config(**config)
    if args.run_id:
        config.run_id = args.run_id
    logger.info(f"Configuration loaded: {config}")
    main(config, args.no_preeval)
