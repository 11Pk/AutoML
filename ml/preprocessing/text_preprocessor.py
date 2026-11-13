
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer



class SentenceEmbeddingTransformer(
    BaseEstimator, TransformerMixin
):

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(
            self.model_name
        )
        return self

    def transform(self, X):

        texts = (
            pd.Series(X)
            .fillna("")
            .astype(str)
            .tolist()
        )

        return self.model_.encode(
            texts,
            show_progress_bar=False
        )

class TextColumnExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return (
            pd.DataFrame(X)
            .iloc[:, 0]
            .fillna("")
            .astype(str)
            .tolist()
        )


class TextStatistics(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        texts = pd.Series(X).fillna("").astype(str)

        result = pd.DataFrame({
            "character_count": texts.str.len(),
            "word_count": texts.str.split().str.len(),
            "average_word_length": (
                texts.str.len() /
                texts.str.split().str.len().replace(0, 1)
            ),
            "sentence_count": texts.str.count(r"[.!?]+")
        })

        return result.to_numpy()


def build_text_pipeline(method):

    if method == "tfidf":

        transformer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )

        return Pipeline([
            ("extract", TextColumnExtractor()),
            ("vectorizer", transformer)
        ])

    elif method == "count":

        transformer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            stop_words="english"
        )

        return Pipeline([
            ("extract", TextColumnExtractor()),
            ("vectorizer", transformer)
        ])

    elif method == "text_stats":

        return Pipeline([
            ("extract", TextColumnExtractor()),
            ("statistics", TextStatistics()),
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])
    elif method == "embeddings":

        return Pipeline([
            ("extract", TextColumnExtractor()),
            ("embedding", SentenceEmbeddingTransformer())
        ])

    else:

        raise ValueError(
            f"Unsupported text method: {method}"
        )
