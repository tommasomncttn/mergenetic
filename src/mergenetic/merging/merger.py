from abc import ABC, abstractmethod
import shutil
from pathlib import Path
import yaml
import torch
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from mergenetic import clean_gpu
from typing import Any

import logging

logger = logging.getLogger(__name__)


class Merger(ABC):
    def __init__(
        self,
        run_id: str,
        path_to_store_yaml: str,
        path_to_store_merged_model: str,
        dtype: str,
        *args: Any,
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
        *args :
            is the list of arguments that will be used to create the configuration file.
        **kwargs :
            is the list of keyword arguments that will be used to create the configuration file.

        Returns
        -------
        None
        """

        # init ABC (parent class)
        super().__init__()

        # storing all the paths
        self.path_to_store_yaml = Path(path_to_store_yaml) / run_id / "config.yaml"
        self.path_to_store_merged_model = Path(path_to_store_merged_model) / run_id

        # storing the data type
        self.dtype = dtype

    def check_and_delete_yaml(self):
        """
        Method to access and delete the yaml configuration file from the local storage. It is used to remove the yaml configuration file before creating a new one.

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
        Method to access and delete the merged model from the local storage. It is used to remove the merged model before creating a new one.

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

    def merge_model_from_configuration(
        self,
        path_to_yaml: Path | str,
        cuda: str | None = None,
        copy_tokenizer: bool = True,
        lazy_unpickle: bool = True,
        low_cpu_memory=True,
    ) -> Path | str:
        """
         Abstract method to merge the models based on the configuration file created for merging models according to a given technique (DARE, TIES, etc.) mergekit implementation.
         It requires the path to the yaml configuration file created.
         It merges the models based on this info and class attributes.
         store the merged model in the folder specified in the class attributes.

         Parameters
         ----------
         path_to_yaml : Path or str
             is the path to the yaml configuration file created.
         cuda : str or None
             is the device to be used for merging the models.
         copy_tokenizer : bool
             is the flag to copy the tokenizer (mergekit configurations).
         lazy_unpickle : bool
             is the flag to use lazy unpickle (mergekit configurations).
         low_cpu_memory : bool
             is the flag to use low cpu memory (mergekit configurations).

         Returns
         -------
        path_to_store_merged_model : : Path or str
             is the path to the merged model created.
        """

        # safe clean to avoid out of memory
        clean_gpu()

        # delete the previous merged model if present
        self._delete_merged_model_local()

        # open the configuration file for merging
        with open(path_to_yaml, "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

        if cuda == None:
            cuda = torch.cuda.is_available()

        # init merge options
        logger.info(f"Merging with device: {cuda}")
        options_merging = MergeOptions(
            cuda=cuda,
            copy_tokenizer=copy_tokenizer,
            lazy_unpickle=lazy_unpickle,
            low_cpu_memory=low_cpu_memory,
        )

        # merge the models
        run_merge(
            merge_config,
            out_path=str(self.path_to_store_merged_model),
            options=options_merging,
        )

        # safe clean to avoid out of memory
        clean_gpu()

        # return the path to the merged model
        return self.path_to_store_merged_model

    @abstractmethod
    def create_individual_configuration(self, *args, **kwargs) -> Path | str:
        """
        Abstract method to create a configuration file for merging models according to a given technique (DARE, TIES, etc.) mergekit implementation.
        It requires dynamic information specific to each merging operation (e.g., weights, densities, etc.).
        It creates the configuration file based on this info and class attributes.
        store the configuration file in the folder specified in the class attributes.

        Parameters
        ----------
        *args :
            is the list of arguments that will be used to create the configuration file.
        **kwargs :
            is the list of keyword arguments that will be used to create the configuration file.

        Returns
        -------
        Path or str
            is the path to the yaml configuration file created.
        """
        pass
