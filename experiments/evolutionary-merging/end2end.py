# import required modules
import argparse
import logging

import numpy as np
import pandas as pd
import yaml
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from mergenetic.evaluation.utils import evaluate_models, retrieve_thetas
from mergenetic.merging import SlerpMerger
from mergenetic.optimization.predefined_problems import (
    ConfigPE,
    CrossLingualMathProblem,
)
from mergenetic.searcher import Searcher
from mergenetic.utils import Config

logger = logging.getLogger(__name__)


def main(config: Config):

    # STEP 1. Load the data
    lang_id = config.langs[0]
    df = pd.read_csv(config.datasets[lang_id])
    logger.info("STEP 1 completed: Data loaded")

    # STEP 2. Extract random anchor points
    anchors = df.sample(n=config.n_samples, random_state=config.seed).index
    anchors_weights = np.ones(len(anchors)) / len(anchors)
    anchors, anchors_weights
    logger.info("anchors: %s", anchors)
    logger.info("STEP 2 completed: Anchors extracted")

    # STEP 3. Get the responses of the base models
    _ = evaluate_models(config)
    logger.info("STEP 3 completed: Predictions obtained")

    # STEP 4. Get the thetas
    thetas = retrieve_thetas(config)
    logger.info(f"STEP 4 completed: Thetas obtained: {thetas}")

    # STEP 5. Extract the samples using the anchors as index and the test set without those samples
    sampled_df = df.loc[anchors].copy()
    test_df = df.drop(anchors)

    logger.info("Sampled DataFrame: %s", sampled_df)
    logger.info("STEP 5 completed: Samples extracted")

    # STEP 6. Unpack some parameters and set the accuracy estimation parameters
    thetas_list = list(thetas.values())
    pop_size = config.pop_size
    n_iter = config.n_iter
    run_id = config.run_id
    bench = config.bench
    mode = config.mode
    est_parameters = ConfigPE(
        thetas=thetas_list,
        weights=anchors_weights,
        sample_ids=anchors,
        bench=bench,
        mode=mode,
    )

    logger.info(
        f"STEP 6 completed: Performance estimation parameters set: {est_parameters}"
    )

    # STEP 7. Define the merger
    path_to_store_yaml = f"{config.path_to_store_config}/{run_id}"
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

    device = config.device if config.device else "cuda"
    logger.info("STEP 7 completed: Merger defined")

    # STEP 8. Define the problem
    problem = CrossLingualMathProblem(
        merger,
        test_df=test_df,
        search_df=sampled_df,
        lang_id=lang_id,
        conf_pe=est_parameters,
        device=device,
        n_var=11,
        n_obj=1,
        n_eq_constr=0,
        n_ieq_constr=0,
        discrete=True,
        load_in_4bit=True,
        eval_batch_size=config.eval_batch_size,
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
    results_path = f"{config.path_to_store_config}/{run_id}"
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
    searcher.test()

    logger.info("Experiment completed, results stored in: %s", results_path)


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

    config = Config(**config)
    if args.seed:
        config.seed = args.seed
        print(f"Overwriting seed with arg: {args.seed}")

    if args.device:
        config.device = args.device
        print(f"Overwriting device with arg: {args.device}")

    if args.run_id:
        config.run_id = args.run_id
        print(f"Overwriting run_id with arg: {args.run_id}")

    if args.n_samples:
        config.n_samples = args.n_samples
        print(f"Overwriting n_samples with arg: {args.n_samples}")

    print(f"Final configuration: {config}")

    main(config)
