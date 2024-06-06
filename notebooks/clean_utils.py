"""
Utilities for preprocessing text data.
"""
import re
from string import whitespace


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to a single space."""
    pattern = '[' + whitespace + ']+'
    return re.sub(pattern, ' ', text)


def remove_punctuation(text: str) -> str:
    """Remove the punctuation from text.

    There are definitely some problems with removing all punctuation.
    For example, $1,234.61 -> 123461.
    Also, 
        @ helps identify email addresses
        :// helps identify URLs

    Also, some of the single and double quotes aren't being removed.

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
    """Clean text by applying the set of functions defined above."""
    
    # Initialize return value.
    cleaned_text = text

    # Apply all cleaning functions.
    cleaned_text = cleaned_text.lower()
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = collapse_whitespace(cleaned_text)

    return cleaned_text


if __name__ == '__main__':
    print(__doc__)
