"""A preprocessor."""
import pandas as pd
import re

from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation

class Preprocessor(BaseEstimator, TransformerMixin):
    @staticmethod
    def strip_html(X):
        """Remove HTML tags from a corpus X."""
        return [re.sub(r'<[^>]+>', ' ', doc) for doc in X]

    @staticmethod
    def strip_forward_slashes(X):
        """Replace / by a space in a corpus X."""
        return [doc.replace('/', ' ') for doc in X]

    @staticmethod
    def strip_punctuation(X):
        """
        Remove punctuation from a corpus X.
        """
        def strip_punc_from_doc(doc: str) -> str:
            """Remove punctuation from a single document doc."""
            doc_trans = doc
            for p in punctuation:
                doc_trans = doc_trans.replace(p, '')
            return doc_trans

        return [strip_punc_from_doc(doc) for doc in X]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_list()

        X_trans = X
        X_trans = [doc.lower() for doc in X_trans]
        X_trans = Preprocessor.strip_html(X_trans)
        X_trans = Preprocessor.strip_forward_slashes(X_trans)
        X_trans = Preprocessor.strip_punctuation(X_trans)

        return X_trans

if __name__ == '__main__':
    print(__doc__)
