import re
from typing import Iterable, List

from nltk.tokenize import wordpunct_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Simple text preprocessing transformer for sklearn pipelines."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[str]) -> List[List[str]]:
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if self.lowercase:
            text = text.lower()

        tokens = wordpunct_tokenize(text)

        if self.remove_punctuation:
            tokens = [token for token in tokens if re.search(r"\w", token)]

        return tokens


def preprocess_text(text: str) -> List[str]:
    """Lowercase, tokenize, and remove punctuation from a single string."""
    return TextPreprocessor()._preprocess_text(text)


def build_preprocessing_pipeline() -> Pipeline:
    """Build an sklearn-compatible text preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("text_preprocessor", TextPreprocessor()),
        ]
    )
