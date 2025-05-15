import datasets

SEED = 42


def process_docs_sample(dataset: datasets.Dataset) -> datasets.Dataset:
    # sample from the dataset 20 examples
    dataset = dataset.add_column("idx", list(range(len(dataset))))
    return dataset


def process_docs_test(dataset: datasets.Dataset) -> datasets.Dataset:
    # sample from the dataset 100 examples
    dataset = dataset.add_column("idx", list(range(len(dataset))))
    return dataset
