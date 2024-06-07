"""
Tokenize documents, remove English stop words, and stem the results.
"""
import pickle
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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
    """Stems each word of tokens."""
    return words


def to_words_text(docs: dict) -> dict:
    """Parition to words every entry of a set of docs."""
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


# def apply_to_column_elements(f, x: dict) -> dict:
#     for label in x:
#         column = x[label]
#         x[label] = [f(element) for element in column]
#     return x


# def main():

#     dataset = 'yelp_review_full'
#     docs_all = load(dataset)

#     # strings needed to specify output filenames
#     dir = 'data'
#     tag_suffix = 'tokenized.pkl'

#     df_tokenized = {}
#     for (key, docs) in docs_all.items():
#         new_dict = docs
#         new_dict = to_words_text(new_dict)
#         new_dict = remove_stop_words_text(new_dict)
#         new_dict = stem_text(new_dict)        
#         df_tokenized[key] = new_dict
        
#         filename = f'{dir}/{dataset}-{key}-{tag_suffix}'
#         with open(filename, 'wb') as f:
#             pickle.dump(df_tokenized[key], f)     


if __name__ == '__main__':
    print(__doc__)
