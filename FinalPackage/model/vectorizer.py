from sklearn.base import TransformerMixin, BaseEstimator


class VectorizerWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, model, yFlag=True):
        self.model = model
        self.yFlag = yFlag

    def fit(self, *args):
        if self.yFlag is True:
            self.model.fit(args[0], args[1])
        else:
            self.model.fit(args[0])
        return self

    def transform(self, *args):
        return {"sparseX": self.model.transform(args[0]), "vocab": self.model.vocabulary_}


class WordVectorWrapper:
    def __init__(self, model):
        self.model = model

    def getWordVectors(self, words):
        result = self.model.query(words)
        return result

