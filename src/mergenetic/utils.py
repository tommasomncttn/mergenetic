import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    A class required to run the evolve search. It allows to set the parameters for the evolution.
    """

    # Required parameters
    pop_size: int = None
    n_iter: int = None
    path_to_store_config: str = None
    path_to_store_merged_model: str = None
    datasets: Dict[str, str] = None
    run_id: str = None
    bench: str = None
    models: Dict[str, str] = None
    task_type: str = None  # FG_MATH or MULTIPLECHOICE
    langs: List[str] = None

    # Optional parameters
    seed: int = 420
    n_samples: int = 20
    mode: str = "gmpirt"
    dtype: str = "bfloat16"
    device: Optional[str] = "cuda:1"
    base_model: Optional[Any] = None
    force_evaluation: bool = False
    eval_batch_size: int = 64


@dataclass
class ConfigLmEval(Config):
    """
    A class required to run the evolve search.
    It allows to set the parameters for the evolution and evaluation with lm-eval-harness.
    """

    metric: Optional[Any] = None
    tasks: Dict[str, Dict[str, Dict[str, str]]] = None
    additional_templates_folder: Optional[str] = None
    load_in_4bit: bool = True
    eager_mode: bool = False


@dataclass
class ConfigMultiObjective:
    """
    **Only for LM-EVAL.**
    A class required to run the evolve search in a generic multi-objective setting.
    It allows to set the parameters for the evolution and evaluation with lm-eval-harness.
    """

    # Required parameters
    run_id: str = None
    n_iter: int = None
    pop_size: int = None
    metric: Optional[Any] = None
    models: List[str] = None
    path_to_store_config: str = None
    path_to_store_merged_model: str = None
    additional_templates_folder: Optional[str] = None
    tasks: List[str] = None

    # Optional parameters
    seed: int = 420
    n_samples: int = 20
    dtype: str = "bfloat16"
    eval_batch_size: int = 32
    device: Optional[str] = "cuda"
    base_model: Optional[Any] = None
    force_evaluation: bool = False
    load_in_4bit: bool = True
    eager_mode: bool = False


############################################################################################################
# UTILS 4 MODELS GENERATION AND MANAGEMENT                                                                 #
############################################################################################################


def get_batched_model_predictions(
    model,
    tokenizer,
    df,
    batch_size=32,
    randomness=False,
    device="cuda",
    max_token=1024,
    print_output=False,
    apply_chat=False,
    dataset_nm=None,
    custom_prompt_template=False,
):

    # Check if prompt column is available in the DataFrame
    assert "prompt" in df.columns, "prompt column not found in the dataframe"

    if custom_prompt_template:
        df["prompt"] = df["prompt"].apply(
            lambda x: custom_prompt_template.format(input=x)
        )

    else:
        if not apply_chat or ("arc_test" in dataset_nm):
            df["prompt"] = df["prompt"].apply(lambda x: f"Question: {x}\nAnswer:")

    # Transform data into list of questions
    list_data = df["prompt"].tolist()

    # Batch the list of questions
    assert batch_size > 0, "batch size must be greater than 0"
    batched_list_data = [
        list_data[i : i + batch_size] for i in range(0, len(list_data), batch_size)
    ]

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    final_answers = []

    # Disable gradient calculation to reduce memory usage
    with torch.no_grad():
        # Loop over the batches
        for batch in tqdm(batched_list_data, desc="Processing batches"):
            # Prepare model inputs
            questions = batch  # Directly use the prompts
            # Tokenize inputs
            if apply_chat:
                # Prepare model inputs
                questions = [
                    [{"role": "user", "content": prompt}] for prompt in questions
                ]
                questions = [
                    tokenizer.apply_chat_template(
                        prompt, tokenize=False, system_message=""
                    )
                    for prompt in questions
                ]

            logger.info(f"First question after processing: {questions[0]}")

            model_inputs = tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_token,
            ).to(device)

            # check that there are no NaNs in the model inputs
            if torch.isnan(model_inputs.input_ids).any():
                raise ValueError("NaNs in model inputs input IDs")

            if torch.isnan(model_inputs.attention_mask).any():
                raise ValueError("NaNs in model inputs attention mask")
            try:
                # Generate answers
                if custom_prompt_template:
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_token,
                        temperature=0,  # Control randomness
                        top_p=1.0,  # Control randomness
                        repetition_penalty=1.0,  # Control repetition
                        eos_token_id=tokenizer.eos_token_id,
                    )

                else:
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_token,
                        do_sample=randomness,  # Control randomness
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode generated IDs to get answers
                answers = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Error during model generation: {e}")
                answers = ["GENERATION FAILED"] * len(questions)
            # Append answers to final list
            final_answers.extend(answers)

            # Optionally print prompts and their corresponding answers
            if print_output:
                for prompt, answer in zip(batch, answers):
                    logger.debug(f"Prompt: {prompt}\nAnswer: {answer}\n---")

            # Clean up to free memory

            model_inputs = model_inputs.to("cpu")
            try:
                generated_ids = generated_ids.to("cpu")
            except (
                AttributeError
            ):  # Handle cases where generated_ids might not have .to()
                pass
            torch.cuda.empty_cache()
    logger.debug(final_answers)
    return final_answers


# cleaner for gpu
def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()


# loading model on automatically chosen device
def loading_model_tokenizer_gpu_only_4bit_local(path):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is off :(")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.debug(f"Loading model from: {path}")

    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cuda:0", quantization_config=quant_config
    )
    model.config.use_cache = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

    return model, tokenizer


############################################################################################################
# UTILS 4 TINYBENCHMARK: https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/tutorials/utils.py     #
############################################################################################################


def prepare_data(scenarios, data):
    """
    Prepare the data by determining the positions of items within each scenario and subscenario.

    Parameters:
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.

    Returns:
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    """

    i = 0
    subscenarios_position = {}

    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in scenarios.keys():
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data["data"][sub]["correctness"].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1

    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in scenarios.keys():
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position


def create_responses(scenarios, data):
    """
    Create a matrix of responses for the chosen scenarios.

    Parameters:
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.

    Returns:
    - A numpy array of responses for the chosen scenarios.
    """

    responses = [
        np.vstack([data["data"][sub]["correctness"] for sub in scenarios[scenario]]).T
        for scenario in scenarios.keys()
    ]
    responses = np.hstack(responses)
    return responses


scenarios = {
    "harness_truthfulqa_mc_0": ["harness_truthfulqa_mc_0"],
    "gsm8k": ["harness_gsm8k_5"],
    "winogrande": ["harness_winogrande_5"],
    "arc": ["harness_arc_challenge_25"],
    "hellaswag": ["harness_hellaswag_10"],
    "mmlu": [
        "harness_hendrycksTest_abstract_algebra_5",
        "harness_hendrycksTest_anatomy_5",
        "harness_hendrycksTest_astronomy_5",
        "harness_hendrycksTest_business_ethics_5",
        "harness_hendrycksTest_clinical_knowledge_5",
        "harness_hendrycksTest_college_biology_5",
        "harness_hendrycksTest_college_chemistry_5",
        "harness_hendrycksTest_college_computer_science_5",
        "harness_hendrycksTest_college_mathematics_5",
        "harness_hendrycksTest_college_medicine_5",
        "harness_hendrycksTest_college_physics_5",
        "harness_hendrycksTest_computer_security_5",
        "harness_hendrycksTest_conceptual_physics_5",
        "harness_hendrycksTest_econometrics_5",
        "harness_hendrycksTest_electrical_engineering_5",
        "harness_hendrycksTest_elementary_mathematics_5",
        "harness_hendrycksTest_formal_logic_5",
        "harness_hendrycksTest_global_facts_5",
        "harness_hendrycksTest_high_school_biology_5",
        "harness_hendrycksTest_high_school_chemistry_5",
        "harness_hendrycksTest_high_school_computer_science_5",
        "harness_hendrycksTest_high_school_european_history_5",
        "harness_hendrycksTest_high_school_geography_5",
        "harness_hendrycksTest_high_school_government_and_politics_5",
        "harness_hendrycksTest_high_school_macroeconomics_5",
        "harness_hendrycksTest_high_school_mathematics_5",
        "harness_hendrycksTest_high_school_microeconomics_5",
        "harness_hendrycksTest_high_school_physics_5",
        "harness_hendrycksTest_high_school_psychology_5",
        "harness_hendrycksTest_high_school_statistics_5",
        "harness_hendrycksTest_high_school_us_history_5",
        "harness_hendrycksTest_high_school_world_history_5",
        "harness_hendrycksTest_human_aging_5",
        "harness_hendrycksTest_human_sexuality_5",
        "harness_hendrycksTest_international_law_5",
        "harness_hendrycksTest_jurisprudence_5",
        "harness_hendrycksTest_logical_fallacies_5",
        "harness_hendrycksTest_machine_learning_5",
        "harness_hendrycksTest_management_5",
        "harness_hendrycksTest_marketing_5",
        "harness_hendrycksTest_medical_genetics_5",
        "harness_hendrycksTest_miscellaneous_5",
        "harness_hendrycksTest_moral_disputes_5",
        "harness_hendrycksTest_moral_scenarios_5",
        "harness_hendrycksTest_nutrition_5",
        "harness_hendrycksTest_philosophy_5",
        "harness_hendrycksTest_prehistory_5",
        "harness_hendrycksTest_professional_accounting_5",
        "harness_hendrycksTest_professional_law_5",
        "harness_hendrycksTest_professional_medicine_5",
        "harness_hendrycksTest_professional_psychology_5",
        "harness_hendrycksTest_public_relations_5",
        "harness_hendrycksTest_security_studies_5",
        "harness_hendrycksTest_sociology_5",
        "harness_hendrycksTest_us_foreign_policy_5",
        "harness_hendrycksTest_virology_5",
        "harness_hendrycksTest_world_religions_5",
    ],
}
