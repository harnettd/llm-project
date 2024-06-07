"""
Load, clean, and dump to CSV the imdb dataset.
"""
import datasets
import pandas as pd

from clean_utils import clean


def load(ds_name: str) -> dict:
    """
    Load train and test data from a dataset.

    :param ds_name: The name of the dataset
    :type ds_name: str

    :return: train and test DataFrames
    :rtype: dict[pd.DataFrame, pd.DataFrame]
    """
    dataset = datasets.load_dataset(ds_name, trust_remote_code=True)

    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    return {
        'train': df_train, 
        'test': df_test, 
    }


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the first column (text) of df.

    For each text document, lowercase the text, collapse whitespace, and
    strip punctuation. 
    """
    df_copy = df.copy()
    df_copy['text'] = df_copy['text'].transform(clean)
    return df_copy


if __name__ == '__main__':
    print(__doc__)
