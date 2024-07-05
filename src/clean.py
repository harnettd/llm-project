"""Load, clean, and dump to CSV the imdb dataset."""
import datasets
import pandas as pd
import re

from string import whitespace


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


def collapse_whitespace(text: str) -> str:
    """
    Collapse all whitespace sequences to a single space.
    
    Usage examples:
    >>> collapse_whitespace(' \n \t   \n\t   ')
    ' '
    """
    pattern = '[' + whitespace + ']+'
    return re.sub(pattern, ' ', text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text.

    :param text: Text to be stripped of punctuation
    :type text: str

    :return: Text with punctuation removed
    :rtype: str

    Usage examples:
    >>> remove_punctuation('a[](){}b #$%&* c/:;<=>?d')
    'ab  cd'
    """
    punctuation = r'''[][(){}\'"!#$%&*+,-./:;<=>?@^_`|~]+'''
    return re.sub(punctuation, '', text)


def clean(text: str) -> str:
    """In text, lower-case, collapse whitespace, and strip punctuation. """
    # Initialize return value.
    cleaned_text = text

    # Apply all cleaning functions.
    cleaned_text = cleaned_text.lower()
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = collapse_whitespace(cleaned_text)

    return cleaned_text


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
