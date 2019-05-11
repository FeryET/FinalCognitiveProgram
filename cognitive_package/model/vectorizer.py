from sklearn.base import TransformerMixin, BaseEstimator
from pymagnitude import Magnitude


class VectorizerWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, model, yFlag=True):
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
    def __init__(self, model):
        self.model = model

    def getWordVectors(self, words):
        result = self.model.query(words)
        return result

