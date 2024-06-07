"""
Tokenize documents, remove English stop words, and stem the results.
"""
from nltk import PorterStemmer
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ps = PorterStemmer()

def load(dataset: str) -> dict:
    """
    Load cleaned train and test CSV files to DataFrames.

    :return: train and test dictionaries
    :rtype: dict[dict, dict]
    """
    dir = '../data'
    tag_suffix = 'cleaned.csv'

    df_train = pd.read_csv(f'{dir}/{dataset}-train-{tag_suffix}')
    df_test = pd.read_csv(f'{dir}/{dataset}-test-{tag_suffix}')

    return {
        'train': df_train.to_dict(orient='list'), 
        'test': df_test.to_dict(orient='list'), 
    }


def to_words(text: str) -> list:
    """Split text into words on whitespace."""
    try:
        split_text = text.split()
    except AttributeError:
        return text
    
    return text.split()


def remove_stop_words(words: list) -> list:
    """Remove English stop words from a list of tokens."""
    for word in words:
        if word in ENGLISH_STOP_WORDS:
            words.remove(word)
    return words


def stem(words: str) -> str:
    """Stem each word of tokens."""
    for (idx, word) in enumerate(words):
        stemmed_word = ps.stem(word)
        if stemmed_word != word:
            words[idx] = stemmed_word
    return words


def to_words_text(docs: dict) -> dict:
    """Parition into words every entry of a dict of docs."""
    docs_copy = docs.copy()
    docs_copy['text'] = [to_words(text) for text in docs_copy['text']]
    return docs_copy


def remove_stop_words_text(docs: dict) -> dict:
    """Remove English stop words from every entry of docs."""
    docs_copy = docs.copy()
    docs_copy['text'] = [remove_stop_words(text) for text in docs_copy['text']]
    return docs_copy


def stem_text(docs: dict) -> dict:
    """Apply stemming to the words of every entry of df."""
    docs_copy = docs.copy()
    docs_copy['text'] = [stem(text) for text in docs_copy['text']]
    return docs_copy


if __name__ == '__main__':
    print(__doc__)
