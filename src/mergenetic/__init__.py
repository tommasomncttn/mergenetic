import logging
import os
from pathlib import Path

import git

logger = logging.getLogger(__name__)

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

try:
    CACHE_DIR = (
        Path(git.Repo(Path.cwd(), search_parent_directories=True).working_dir) / "cache"
    )
except git.exc.InvalidGitRepositoryError as e:
    # should use the predefined hugging-face directory
    logger.error(e)
    CACHE_DIR = None
    logger.debug("\n setting default hugging-face cache directory")

if "mergenetic" not in str(PROJECT_ROOT):

    logger.warning(f"PROJECT_ROOT IS HARCODED AT {PROJECT_ROOT}")
    CACHE_DIR = PROJECT_ROOT / "cache"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["CACHE_DIR"] = str(CACHE_DIR)

#############################################
# importing the necessary modules to set up the project
#############################################
from .utils import *
