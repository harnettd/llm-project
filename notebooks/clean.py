"""
Load, preprocess, and dump to CSV multi_news dataset.
"""
import datasets
import pandas as pd

from clean_utils import clean


def load_imdb() -> dict:
    """
    Load data from the imdb dataset.

    :return: train, test, and unsupervised DataFrames
    :rtype: tuple[pd.DataFrame]
    """
    dataset = datasets.load_dataset('imdb', trust_remote_code=True)

    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    df_unsupervised = pd.DataFrame(dataset['unsupervised'])

    return {
        'train': df_train, 
        'test': df_test, 
        'unsupervised': df_unsupervised
    }


def load_yelp() -> dict:
    """
    Load data from the Yelp reviews set dataset.

    :return: train, test, and unsupervised DataFrames
    :rtype: tuple[pd.DataFrame]
    """
    dataset = datasets.load_dataset('yelp_review_full', trust_remote_code=True)

    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    return {
        'train': df_train, 
        'test': df_test
    }


def load_news() -> dict:
    """
    Load data from the newsgroups dataset.

    :return: train and test DataFrames
    :rtype: tuple[pd.DataFrame]
    """
    dataset = datasets.load_dataset('SetFit/20_newsgroups', trust_remote_code=True)

    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    return {
        'train': df_train, 
        'test': df_test
    }


def load(dataset_name: str)-> dict:
    """
    Load a dataset by name.
    """
    if dataset_name == 'imdb':
        return load_imdb()
    
    if dataset_name == 'yelp_review_full':
        return load_yelp()
    
    if dataset_name == 'SetFit/20_newsgroups':
        return load_news()

    raise FileNotFoundError


def drop_short_docs(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Return a DataFrame with samples with short strings dropped.
    
    :param df: A DataFrame with string entries
    :param threshold: Entries with characters less than this are dropped 
    """
    return df[(df.map(len) > threshold).all(axis='columns')]


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the first column (text) of df."""
    df_copy = df.copy()
    df_copy['text'] = df_copy['text'].transform(clean)
    return df_copy


def main() -> None:
    # the name of the dataset to be loaded and cleaned
    dataset_name = 'SetFit/20_newsgroups'
    df_dict = load(dataset_name)
    
    # strings needed to specify output filenames
    dir = 'data'
    tag_suffix = 'cleaned.csv'

    # Clean and write to CSV all three DataFrames.
    for (key, df) in df_dict.items():
        df_clean = df.pipe(clean_text)
            #.pipe(drop_short_docs, threshold=100)\

        filename = f'{dir}/{dataset_name}-{key}-{tag_suffix}'
        df_clean.to_csv(
            filename, 
            sep=',', 
            header=True, 
            index=False
        )  


if __name__ == '__main__':
    main()
