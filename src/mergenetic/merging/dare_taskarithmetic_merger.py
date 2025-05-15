import logging
from pathlib import Path
from typing import Iterable

import yaml

from mergenetic.merging.merger import Merger

logger = logging.getLogger(__name__)


class DareTaskArithmeticMerger(Merger):
    def __init__(
        self,
        run_id: str,
        path_to_base_model: str,
        model_paths: list,
        path_to_store_yaml: str,
        path_to_store_merged_model: str,
        dtype: str,
    ) -> None:
        """
        Concrete class for merging models. It is used to create a configuration file for merging models according to Linear.
        It requires all the information for merging through mergekit library according to the given merging technique.
        Two type of information are used:
            1. Static information common to all merged models with that technique (e.g., base model, dtype, etc.); these are passed through the class init.
            2. Dynamic information specific to each merging operation (e.g., weights, densities, etc.); these are passed through the method for merging.

        Parameters
        ----------
        run_id : str
            is the id of the run that will be used to store all the yaml configuration files for merging.
        path_to_base_model : str
            is the path to the downloaded base model that will be used for merging.
        model_paths : list
            variable number of paths for the models that will be merged.
        path_to_store_yaml : str
            is the path to the folder where the yaml configuration files for merging will be stored.
        path_to_store_merged_model : str
            is the path to the folder where the merged models will be stored.
        dtype : str
            is the data type of the models (e.g., float16, float32, etc.).

        Returns
        -------
        None
        """
        super().__init__(run_id, path_to_store_yaml, path_to_store_merged_model, dtype)

        # store the paths to the models for ties dare
        self.path_to_base_model = Path(path_to_base_model)
        self.model_paths = [Path(path) for path in model_paths]

    def create_individual_configuration(self, weights: Iterable) -> Path | str:
        """
        Method to create a configuration file for merging models according to DARE_TIES mergekit implementation.
        It requires dynamic information specific to each merging operation (e.g., weights, densities, etc.).
        It creates the configuration file based on this info and class attributes.
        Store the configuration file in the folder specified in the class attributes.

        Parameters
        ----------
        weights_and_densities : Iterable
            list of pairs of weights and densities for each model.

        Returns
        -------
        Path or str
            is the path to the yaml configuration file created.
        """
        # Create a list of dictionaries for the models and their parameters
        weights = [float(weight) for weight in weights]

        # Create a dictionary for the configuration file
        model_info = [{"model": str(self.path_to_base_model)}]
        for model_path, weight in zip(self.model_paths, weights):
            model_info.append(
                {"model": str(model_path), "parameters": {"weight": weight}}
            )

        config = {
            "models": model_info,
            "merge_method": "dare_linear",
            "base_model": str(self.path_to_base_model),
            "parameters": {"int8_mask": True},
            "dtype": self.dtype,
        }

        # Create Directory for YAML file
        config_directory = Path(self.path_to_store_yaml).parent
        config_directory.mkdir(parents=True, exist_ok=True)

        # Write the dictionary to a YAML file
        with open(self.path_to_store_yaml, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            logger.info(f"Configuration file created at {self.path_to_store_yaml}")

        # Return the path to the YAML file
        return Path(self.path_to_store_yaml)
