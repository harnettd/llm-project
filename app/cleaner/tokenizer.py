"""A tokenizer."""
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, stemmer, stop_words=None):
        """
        Initialize a Tokenizer.

        :param stemmer: An instance of a word stemmer
        :param stop_words: A list of words to remove from the corpus
        """
        self.stemmer = stemmer
        self.stop_words = stop_words

    def remove_stop_words(self, X):
        """
        Remove stop words from a corpus X.
        """
        def remove_stop_words_from_doc(doc):
            """Remove stop words from a document doc."""
            doc_trans = []
            for word in doc:
                if word not in self.stop_words:
                    doc_trans.append(word)
            return doc_trans

        return [remove_stop_words_from_doc(doc) for doc in X]

    def stem(self, X):
        """
        Stem a corpus X.
        """
        def stem_doc(doc):
            """Stem a document doc."""
            return_doc = []
            for word in doc:
                return_doc.append(self.stemmer.stem(word))
            return return_doc

        return [stem_doc(doc) for doc in X]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_list()

        X_trans = X
        X_trans = [doc.split() for doc in X_trans]
        X_trans = self.remove_stop_words(X_trans)
        X_trans = self.stem(X_trans)
        X_trans = [' '.join(doc) for doc in X_trans]

        return X_trans
    
if __name__ == '__main__':
    print(__doc__)
