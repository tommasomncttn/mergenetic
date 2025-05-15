import os
import pickle
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
from lm_eval.models.vllm_causallms import VLLM

from mergenetic.estimator.utils import estimate_theta_anchors
from mergenetic.evaluation import BaseEvaluator, FGMathEvaluator, MCEvaluator
from mergenetic.evaluation.lm_harness import LmHarnessEvaluator
from mergenetic.utils import (
    Config,
    ConfigLmEval,
    get_batched_model_predictions,
    loading_model_tokenizer_gpu_only_4bit_local,
)

logger = getLogger(__name__)

############################################
# General utility functions                #
############################################


def check_file_exists(file_path: str) -> bool:
    """
    Check if the file exists at the given path
    """
    return Path(file_path).exists()


############################################
# Configuration utility functions          #
############################################


@dataclass
class ConfigEvaluation:
    """
    Configuration Required to request the evaluation for either a mathematical or multiple choice task.
    The dataset, the model, and an output where to save the file must be furnished.
    The file is just a csv with the predictions that can be used to compute the score.
    """

    eval_task: str = ""  # MATH or MULTIPLECHOICE
    dataset_path: str = ""
    model_path: str = ""
    output_path: str = ""
    dtype: str = ""
    eval_batch_size: int = 8


@dataclass
class ConfigThetaEstimation:
    """
    A configuration class to request the estimation of the latent ability of the model.
    Such latent ability are the thetas.
    Tb_path is needed if you wish to use original tinybenchmarks parameters.
    """

    responses_path: str
    bench: str
    output_path: str


@dataclass
class EvalMode(str, Enum):
    RANDOM = "random"
    MPIRT = "mpirt"
    GMPIRT = "gmpirt"
    RC = "rc"
    ADVERSARIAL = "adversarial"
    IDEAL = "ideal"


def validate_evaluation_choice(choice):
    """
    Check if the passed evaluation choice is valid
    """
    try:
        EvalMode(choice)
    except ValueError:
        raise ValueError(
            "Invalid evaluation mode. Choose 'random', 'mpirt', 'gmpirt', or 'rc'."
        )


############################################
# Evaluation functions
############################################
class Tasks(Enum):
    FG_MATH = "FG_MATH"
    MULTIPLECHOICE = "MULTIPLECHOICE"


def task_2_fitness(task: str) -> BaseEvaluator:
    """
    Map the task to the evaluator
    """
    if task == Tasks.FG_MATH.value:
        return FGMathEvaluator
    elif task == Tasks.MULTIPLECHOICE.value:
        return MCEvaluator
    else:
        raise ValueError("Invalid task. Choose 'FG_MATH' or 'MULTIPLECHOICE'.")


def evaluate(
    task: str,
    df: pd.DataFrame,
    model_path: str,
    eval_batch_size: int,
    load_on_gpu: bool,
    get_also_series: bool = True,
) -> float | tuple[float, pd.DataFrame]:

    assert (
        task in Tasks.__members__.keys()
    ), f"Invalid task '{task}'. Choose {Tasks.__members__.keys()}"

    # load the model and tokenizer
    model, tokenizer = loading_model_tokenizer_gpu_only_4bit_local(model_path)

    # get the predictions
    if load_on_gpu:
        device = "cuda:0"
    else:
        device = "cpu"
    df["predictions"] = get_batched_model_predictions(
        model,
        tokenizer,
        df,
        batch_size=eval_batch_size,
        randomness=False,
        device=device,
        print_output=True,
        apply_chat=False,
    )

    # get the correctness
    evaluator: BaseEvaluator = task_2_fitness(task)()
    f = -1 * evaluator.get_correctness(df)

    if get_also_series:
        return [f], evaluator.get_data()
    else:
        return [f]


def evaluate_models(config: Config):
    """
    Evaluate the model on the given task
    """
    if config.datasets:
        # zip together the model paths and output paths
        models = dict(config.models.items())

        langs = config.datasets.keys()
        output_paths = {
            k: config.path_to_store_config + "/" + model.split("/")[-1] + f"_{k}.csv"
            for k, model in config.models.items()
        }

        if config.base_model is not None:
            models["base"] = config.base_model
            output_paths["base"] = (
                config.path_to_store_config
                + "/"
                + config.base_model.split("/")[-1]
                + "_base.csv"
            )
            langs = list(langs) + ["base"]

        evaluations = {}
        for lang in langs:
            model_path = models[lang]
            output_path = output_paths[lang]

            # check if file exists
            if not config.force_evaluation and check_file_exists(output_path):
                logger.info(
                    f"Skipping evaluation for {model_path} as the evaluation file already exists."
                )
                # load the file
                df = pd.read_csv(output_path)
                evaluations[lang] = df
                continue

            logger.info(
                f"Evaluation for {model_path} not found at path {output_path}. Starting evaluation."
            )

            df = pd.read_csv(config.datasets[lang])
            if model_path != "":
                logger.info(f"Head of the dataframe for {lang}: {df.head()}")
                fit, pred_df = evaluate(
                    config.task_type, df, model_path, config.eval_batch_size, True
                )
                if output_path != "":
                    pred_df.to_csv(output_path)
                    logger.info(
                        f"The predictions for {lang} are saved to {output_path}. Fitness: {fit}"
                    )
                evaluations[lang] = pred_df
    else:
        raise ValueError("No datasets provided for evaluation.")

    return evaluations


def evaluate_models_lm_harness(config: ConfigLmEval):
    """
    Evaluate the model on the given search tasks
    """
    tasks = config.tasks["search"]
    langs = config.langs
    output_paths = [
        config.path_to_store_config + "/lm-eval_" + model.split("/")[-1] + f"_{k}.csv"
        for k, model in config.models.items()
    ]
    model_output_langs = zip(output_paths, langs)
    evaluations = {}

    for output_path, lang in model_output_langs:
        # check if file exists
        task = tasks[lang]
        if not config.force_evaluation and check_file_exists(output_path):
            logger.info(
                f"Skipping evaluation for {lang} as the evaluation file already exists."
            )
            # load the file
            df = pd.read_csv(output_path)
            evaluations[lang] = df
            continue

        logger.info(
            f"Evaluation for {lang} not found at path {output_path}. Starting evaluation."
        )

        model_path = config.models[lang]
        model = VLLM(
            pretrained=str(model_path),
            device=config.device,
            dtype=torch.bfloat16,
            quantization="bitsandbytes" if config.load_in_4bit else None,
            gpu_memory_utilization=0.8 if config.load_in_4bit else 0.9,
        )

        evaluator = LmHarnessEvaluator(
            task_name=task,
            sample_ids=None,
            correctness_metric=config.metric,
            lang_id=None,
            is_test=False,
            additional_templates_folder=config.additional_templates_folder,
            batch_size=config.eval_batch_size,
        )
        evaluator.evaluate(model)
        evaluations[lang] = evaluator.get_data()
        evaluations[lang].to_csv(output_path)
        logger.info(f"The predictions for {lang} are saved to {output_path}.")

    if config.base_model is not None:
        out_path = f"{config.path_to_store_config}/lm-eval_{config.base_model.split('/')[-1]}_base.csv"

        if not config.force_evaluation and check_file_exists(out_path):
            logger.info(
                f"Skipping evaluation for the base model as the evaluation file already exists."
            )
            # load the file
            evaluations["base"] = pd.read_csv(out_path)
            return evaluations

        logger.info(
            f"Evaluation for the base model not found at path {out_path}. Starting evaluation."
        )
        base_model = VLLM(
            pretrained=str(config.base_model),
            device=config.device,
            quantization="bitsandbytes" if config.load_in_4bit else None,
            gpu_memory_utilization=0.8 if config.load_in_4bit else 0.9,
        )
        evaluator = LmHarnessEvaluator(
            task_name=tasks["base"],
            sample_ids=None,
            correctness_metric=config.metric,
            lang_id=None,
            is_test=True,
            additional_templates_folder=config.additional_templates_folder,
            batch_size=config.eval_batch_size,
        )
        evaluator.evaluate(base_model)
        evaluations["base"] = evaluator.get_data()
        evaluations["base"].to_csv(out_path)


############################################
# Theta Estimation functions
############################################


def extract_thetas(config: Config):
    """
    Extract thetas for models in ft_models
    """
    responses_paths = {
        l: f"{config.path_to_store_config}/{m.split('/')[-1]}_{l}.csv"
        for l, m in config.models.items()
    }
    output_path_theta_estimation = {
        l: f"{config.path_to_store_config}/{m.split('/')[-1]}_{l}_theta.pkl"
        for l, m in config.models.items()
    }

    if config.base_model is not None:
        responses_paths["base"] = (
            f"{config.path_to_store_config}/{config.base_model.split('/')[-1]}_base.csv"
        )
        output_path_theta_estimation["base"] = (
            f"{config.path_to_store_config}/{config.base_model.split('/')[-1]}_base_theta.pkl"
        )

    thetas = {}

    for l in responses_paths.keys():
        responses_path = responses_paths[l]
        output_path = output_path_theta_estimation[l]

        # init config
        config_theta = ConfigThetaEstimation(
            responses_path=responses_path,
            bench=config.bench,
            output_path=output_path,
        )

        try:
            responses = pd.read_csv(config_theta.responses_path)
        except FileNotFoundError:
            logger.info(f"Responses file not found at {config_theta.responses_path}.")

            logger.info(f"Looking for lm-evaluation-harness evaluation files.")
            model_nm = config.models[l] if l != "base" else config.base_model
            responses_path = f"{config.path_to_store_config}/lm-eval_{model_nm.split('/')[-1]}_{l}.csv"
            responses = pd.read_csv(responses_path)
        y = responses["correctness"].values
        anchors = list(range(len(y)))

        theta = estimate_theta_anchors(y, config_theta.bench, anchors)
        if config_theta.output_path != "":
            if not os.path.exists(config_theta.output_path):
                os.makedirs(os.path.dirname(config_theta.output_path), exist_ok=True)
            with open(config_theta.output_path, "wb") as f:
                pickle.dump(theta, f)
                logger.info(f"Theta saved to {config_theta.output_path}")
        thetas[l] = theta

    return thetas


def retrieve_thetas(config: Config):
    """
    Retrieve thetas from the given path
    """
    thetas_paths = {
        k: f"{config.path_to_store_config}/{m.split('/')[-1]}_{k}_theta.pkl"
        for k, m in config.models.items()
    }

    if config.base_model is not None:
        thetas_paths["base"] = (
            f"{config.path_to_store_config}/{config.base_model.split('/')[-1]}_base_theta.pkl"
        )

    thetas = {}
    for k, path in thetas_paths.items():
        if check_file_exists(path):
            with open(path, "rb") as f:
                thetas[k] = pickle.load(f)
            logger.info(f"Theta for '{k}' retrieved from {path}.")
        else:
            logger.info(f"Theta for '{k}' not found at {path}. Extracting thetas.")
            thetas.update(extract_thetas(config))
            break

    return thetas
