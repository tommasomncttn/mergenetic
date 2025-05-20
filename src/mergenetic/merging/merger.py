import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch  # Ensure torch is imported if used for cuda.is_available
import yaml  # Ensure yaml is imported if used for safe_load
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

from mergenetic import clean_gpu

logger = logging.getLogger(__name__)


class Merger(ABC):
    def __init__(
        self,
        run_id: str,
        path_to_store_yaml: str,
        path_to_store_merged_model: str,
        dtype: str,
        **kwargs: Any,
    ) -> None:
        """
        Abstract class for merging models. It is used to create a configuration file for merging models according to a given technique (DARE, TIES, etc.).
        It requires all the information for merging through mergekit library according to the given merging technique. Two type of information are used:
            1. Static information common to all merged models with that technique (e.g., base model, dtype, etc.); these are passed through the class init.
            2. Dynamic information specific to each merging operation (e.g., weights, densities, etc.); these are passed through the method for merging.

        Parameters
        ----------
        run_id : str
            is the id of the run that will be used to store all the yaml configuration files for merging.
        path_to_store_yaml : str
            is the path to the folder where the yaml configuration files for merging will be stored.
        path_to_store_merged_model : str
            is the path to the folder where the merged models will be stored.
        dtype : str
            is the data type of the models (e.g., float16, float32, etc.).
        **kwargs :
            is the list of keyword arguments that will be used to create the configuration file.

        Returns
        -------
        None
        """

        # init ABC (parent class)
        super().__init__()

        # storing all the paths
        self.run_id = run_id
        # Ensure path_to_store_yaml is the full file path
        self.path_to_store_yaml = Path(path_to_store_yaml) / "config.yaml"
        self.path_to_store_merged_model = Path(path_to_store_merged_model) / self.run_id

        # storing the data type
        self.dtype = dtype
        # Handle other kwargs if necessary, e.g., for specific merger types

    @abstractmethod
    def create_individual_configuration(self, *args: Any, **kwargs: Any) -> Path:
        """
        Creates the individual configuration file for the merger.
        """
        pass

    def check_and_delete_yaml(self) -> None:
        """
        Checks if the YAML configuration file exists and deletes it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.path_to_store_yaml.exists():  # check if the file exists
            self.path_to_store_yaml.unlink()  # Deletes the file
            logger.info(f"Deleted: {self.path_to_store_yaml}")
        else:
            logger.info(f"No file found at: {self.path_to_store_yaml}")

    def _delete_merged_model_local(self) -> None:
        """
        Deletes the merged model from the local storage.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Check if the path exists before attempting to remove it
        if self.path_to_store_merged_model.exists():

            # Use shutil.rmtree to remove the directory
            shutil.rmtree(self.path_to_store_merged_model)
            logger.info(
                f"Deleted folder and all contents: {self.path_to_store_merged_model}"
            )

        else:

            logger.info(f"The folder does not exist: {self.path_to_store_merged_model}")

    def merge_model_from_configuration(self, path_to_yaml: Path) -> Path:
        """
        Merges the model from the configuration file.

        Parameters
        ----------
        path_to_yaml : Path
            is the path to the yaml configuration file created.

        Returns
        -------
        path_to_store_merged_model : : Path or str
             is the path to the merged model created.
        """
        logger.info("Starting model merging process...")
        clean_gpu()
        self._delete_merged_model_local()

        with open(path_to_yaml, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        cfg = MergeConfiguration.model_validate(config_data)

        options = MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=True,
            low_cpu_memory=True,
        )

        logger.debug(f"Running merge with configuration: {cfg}")
        logger.debug(f"Merge options: {options}")

        # Ensure the output path is passed as the second positional argument
        run_merge(cfg, str(self.path_to_store_merged_model), options=options)

        logger.info(f"Model merged successfully at {self.path_to_store_merged_model}")
        clean_gpu()
        return self.path_to_store_merged_model
