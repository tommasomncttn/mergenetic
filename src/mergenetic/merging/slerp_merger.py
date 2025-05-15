import logging
from pathlib import Path
from typing import Iterable

import yaml

from mergenetic.merging.merger import Merger

logger = logging.getLogger(__name__)


class SlerpMerger(Merger):
    def __init__(
        self,
        run_id: str,
        path_to_base_model: Path | str,
        layer_range_base_model: str,
        path_to_model_1: Path | str,
        layer_range_model_1: str,
        path_to_store_yaml: Path | str,
        path_to_store_merged_model: Path | str,
        dtype: str,
    ) -> None:
        """
        Concrete class for merging models. It is used to create a configuration file for merging models according to Slerp.
        It requires all the information for merging through mergekit library according to the given merging technique. Two type of information are used:
            1. Static information common to all merged models with that technique (e.g., base model, dtype, etc.); these are passed through the class init.
            2. Dynamic information specific to each merging operation (e.g., t, layer range); these are passed through the method for merging.

        Parameters
        ----------
        run_id : str
            is the id of the run that will be used to store all the yaml configuration files for merging.
        path_to_base_model : str
            is the path to the downloaded base model that will be used for merging.
        layer_range_base_model : list
            is the list of layer indices for the base model that takes part in the interpolation.
        path_to_model_1 : str
            is the path to the first model that will be merged.
        layer_range_model_1 : list
            is the list of layer indices for the first model that takes part in the interpolation.
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
        # init merger common attributes
        super().__init__(run_id, path_to_store_yaml, path_to_store_merged_model, dtype)

        # store the paths to the models for ties dare
        self.path_to_base_model = Path(path_to_base_model)
        self.path_to_model_1 = Path(path_to_model_1)
        self.layer_range_base_model = layer_range_base_model
        self.layer_range_model_1 = layer_range_model_1

    def create_individual_configuration(self, x: Iterable) -> Path | str:
        """
        Abstract method to create a configuration file for merging models according to SLERP mergekit implementation.
        It requires dynamic information specific to each merging operation (e.g., t, layer range).
        It creates the configuration file based on this info and class attributes.
        store the configuration file in the folder specified in the class attributes.

        Parameters
        ----------
        x : Iterable
            is the list/array of values used to instatiate the configuration file of mergekit for merging through slerp.

        Returns
        -------
        Path | str
            is the path to the yaml configuration file created.

        """

        # convert the array from pymoo to list of float
        x = [float(i) for i in x]

        # extract the values of t for self_attn, mlp, and other
        t_self_attn = x[0:5]
        t_mlp = x[5:10]
        t_other = x[10]

        # create a configuration file from a dictionary
        config = {
            "slices": [
                {
                    "sources": [
                        {
                            "model": str(self.path_to_base_model),
                            "layer_range": self.layer_range_base_model,
                        },
                        {
                            "model": str(self.path_to_model_1),
                            "layer_range": self.layer_range_model_1,
                        },
                    ]
                }
            ],
            "merge_method": "slerp",
            "base_model": str(self.path_to_base_model),
            "parameters": {
                "t": [
                    {"filter": "self_attn", "value": t_self_attn},
                    {"filter": "mlp", "value": t_mlp},
                    {"value": t_other},
                ]
            },
            "dtype": self.dtype,
        }

        # Create Directory for YAML fil
        config_directory = self.path_to_store_yaml.parent
        config_directory.mkdir(parents=True, exist_ok=True)

        # Write the dictionary to a YAML file
        with open(self.path_to_store_yaml, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            logger.info(f"Configuration file created at {self.path_to_store_yaml}")

        # Return the path to the YAML file
        return self.path_to_store_yaml
