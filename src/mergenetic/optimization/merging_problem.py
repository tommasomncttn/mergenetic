from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from lm_eval.models.vllm_causallms import VLLM
from pymoo.core.problem import Problem
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mergenetic.merging.merger import Merger

logger = getLogger(__name__)


class BaseMergingProblem(ABC, Problem):
    """
    Base class for merging problems (single and multi-objective).
    """

    def __init__(
        self,
        merger: Merger,
        n_var: int,
        n_obj: int,
        xl=0,
        xu=1,
        n_eq_constr: int = 0,
        n_ieq_constr: int = 0,  # TODO: verify if needed for single-objective
        eval_batch_size: int = 5,
        discrete: bool = False,
        device: str = "cuda",
        load_in_4bit: bool = True,
        eager_mode: bool = False,
        use_lm_eval: bool = False,  # Flag for using lm-eval-harness (VLLM)
        verbose_evaluation: bool = True,
        test_mode: bool = False,
        custom_prompt_template=False,
    ) -> None:

        self.n_var = n_var
        self.n_obj = n_obj
        self.merger = merger
        self.discrete = discrete
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.use_lm_eval = use_lm_eval  # Enables VLLM
        self.verbose_evaluation = verbose_evaluation
        self.test_mode = test_mode
        self.custom_prompt_template = custom_prompt_template or False
        self.n_eq_constr = n_eq_constr
        self.n_ieq_constr = n_ieq_constr
        self.eager_mode = eager_mode

        if self.load_in_4bit:
            logger.info("Loading models in 4bit quantization...")

        if self.eager_mode:
            logger.info("Using eager mode for VLLM.")

        # Adjust bounds for discrete search
        if self.discrete:
            xl, xu = 0, 10

        # Initialize pymoo problem
        super().__init__(elementwise=True, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        # Tracking performance
        self.objective_list = [f"objective_{i+1}" for i in range(n_obj)]
        self.phenotype_feature_list = [f"genotype_{i+1}" for i in range(n_var)]

        # Single-objective case: one DataFrame
        if self.n_obj == 1:
            self.results_df = pd.DataFrame(
                columns=self.objective_list + self.phenotype_feature_list + ["step"]
            )
        else:
            # Multi-objective case: multiple fitness DataFrames (one per objective)
            self.results_df = {
                obj: pd.DataFrame(
                    columns=[obj] + self.phenotype_feature_list + ["step"]
                )
                for obj in self.objective_list
            }

        self.step = 0

    def _from_array_to_genotype(self, x: np.array) -> Path:
        """Convert an array representation to a model configuration."""
        logger.info(f"Converting genotype to configuration: {x}")
        path_to_config = self.merger.create_individual_configuration(x)
        return self.merger.merge_model_from_configuration(path_to_config)

    def load_model(
        self, path: str, verbose: bool = True
    ) -> VLLM | tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Loads a model and tokenizer using either Hugging Face or VLLM."""
        logger.info(f"Loading model from: {path}")
        logger.info(f"On device: {self.device}. Load in 4bit: {self.load_in_4bit}")

        device = self.device if self.device is not None else "cpu"
        load_on_gpu = "cuda" in device

        if load_on_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU loading was requested.")

        model: AutoModelForCausalLM = None

        if self.use_lm_eval:
            return VLLM(
                pretrained=str(path),
                device=device,
                enforce_eager=self.eager_mode,
                dtype=torch.bfloat16,
                quantization="bitsandbytes" if self.load_in_4bit else None,
                gpu_memory_utilization=0.8 if self.load_in_4bit else 0.9,
                load_format="bitsandbytes" if self.load_in_4bit else "auto",
            )
        else:
            if self.load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path, device_map=device, quantization_config=quant_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(path, device_map=device)

        model.config.use_cache = True
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

        return model, tokenizer

    @abstractmethod
    def metrics_4_genotype(
        self, model, tokenizer=None
    ) -> tuple[list[float], Optional[str]]:
        """Abstract method for evaluating a model."""
        pass

    @abstractmethod
    def test(self):
        """Abstract method for testing the model."""
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Abstract method for getting the data."""
        pass

    def _evaluate(self, x, out, *args, **kwargs):
        """Main evaluation function for optimization."""
        if self.discrete:
            x = x / 10  # Normalize discrete values

        path_to_model = self._from_array_to_genotype(x)
        model = self.load_model(path_to_model)

        logger.info(f"Step {self.step}, evaluating genotype: {x}")
        # Multi-objective fitness handling
        if self.use_lm_eval:
            f, description = self.metrics_4_genotype(model)
        else:
            model, tokenizer = model  # Unpack if not using VLLM
            f, description = self.metrics_4_genotype(model, tokenizer)

        logger.info(f"Model evaluation results: {f}")

        out["F"] = f
        self.step += 1

        log_entry = dict(
            zip(self.phenotype_feature_list, x.flatten()),
            **dict(zip(self.objective_list, f)),
            step=self.step,
        )

        if self.n_obj == 1:
            self.results_df = pd.concat(
                [self.results_df, pd.DataFrame([log_entry])], ignore_index=True
            )
        else:
            # Update each objective's DataFrame separately
            for obj_name, fitness_value in zip(self.objective_list, f):
                # Define the columns for the specific objective's DataFrame
                current_obj_df_columns = self.results_df[obj_name].columns

                # Prepare the new row data, ensuring it only contains the relevant columns.
                # log_entry already contains obj_name mapped to its fitness_value (which is f_i).
                new_row_dict = {col: log_entry[col] for col in current_obj_df_columns}
                new_df_row = pd.DataFrame(
                    [new_row_dict], columns=current_obj_df_columns
                )

                if self.results_df[obj_name].empty:
                    self.results_df[obj_name] = new_df_row
                else:
                    self.results_df[obj_name] = pd.concat(
                        [self.results_df[obj_name], new_df_row],
                        ignore_index=True,
                    )

        del model
        torch.cuda.empty_cache()

        if description:
            logger.debug(f"Individual {x} description: {description}")


class MergingProblem(BaseMergingProblem):
    """
    Implementation of a **single-objective** merging problem.
    """

    def __init__(
        self,
        merger,
        search_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        super().__init__(merger, **kwargs)  # Enforce single-objective
        self.search_df: pd.DataFrame = search_df
        self.test_df: Optional[pd.DataFrame] = test_df

    def get_data(self) -> pd.DataFrame:
        return self.results_df


class MultiObjectiveMergingProblem(BaseMergingProblem):
    """
    Implementation of a multi-objective optimization problem for model merging.

    This class extends BaseMergingProblem to handle multiple competing objectives
    simultaneously (e.g., optimizing for both accuracy and efficiency). Unlike
    single-objective optimization, this creates a Pareto front of solutions
    representing different trade-offs between objectives.

    Each objective requires its own evaluation dataset and metrics. Results are
    tracked separately for each objective, allowing for comprehensive analysis
    of the optimization landscape and trade-offs between competing goals.
    """

    def __init__(
        self,
        merger,
        n_obj: int,
        search_dataframes: Dict[str, pd.DataFrame] | None,
        test_dataframes: Optional[Dict[str, pd.DataFrame]],
        **kwargs,
    ):
        assert (
            n_obj > 1
        ), "MultiObjectiveMergingProblem must have more than one objective!"
        super().__init__(merger, n_obj=n_obj, **kwargs)
        self.search_dataframes: Dict[str, pd.DataFrame] | None = search_dataframes
        self.test_dataframes: Optional[Dict[str, pd.DataFrame]] = test_dataframes

    def get_data(self) -> Dict[str, pd.DataFrame]:
        return self.results_df
