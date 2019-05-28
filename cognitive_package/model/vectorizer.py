from sklearn.base import TransformerMixin, BaseEstimator
from pymagnitude import Magnitude
from gensim.models import FastText
import numpy as np


class VectorizerWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, *args):
        self.model.fit(args[0], args[1])
        return self

    def transform(self, *args):
        return {
            "sparseX": self.model.transform(args[0]),
            "vocab": self.model.vocabulary_,
        }


class WordVectorWrapper:
    MAGNITUDE = "magnitude"
    GENSIM = "gensim"

    def __init__(self, model, magnitude_or_gensim):
        if not (
            magnitude_or_gensim == self.MAGNITUDE
            or magnitude_or_gensim == self.GENSIM
        ):
            raise AttributeError
        self.magnitude_or_gensim = magnitude_or_gensim
        self.model = model

    def getWordVectors(self, words):
        if self.magnitude_or_gensim == self.MAGNITUDE:
            result = self.model.query(words)
        elif self.magnitude_or_gensim == self.GENSIM:
            result = []
            for w in words:
                result.append(self.model[w])
            result = np.array(result)
        return result

