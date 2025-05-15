# download fasttext https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# and save it in models/fasttext/lid.176.bin

import os
from logging import getLogger

import requests

logger = getLogger(__name__)


def prepare_fasttext():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f"{parent_dir}/models/fasttext/lid.176.bin"):
        os.makedirs(f"{parent_dir}/models/fasttext", exist_ok=True)

        with open(f"{parent_dir}/models/fasttext/lid.176.bin", "wb") as f:
            f.write(
                requests.get(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                ).content
            )

        logger.info(
            "fastText model downloaded successfully and saved in models/fasttext/lid.176.bin"
        )
    else:
        logger.info("fastText model already exists")


if __name__ == "__main__":
    prepare_fasttext()
