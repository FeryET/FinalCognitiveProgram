from sklearn.base import BaseEstimator, TransformerMixin


class PipelinePlaceHolder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X = None
        self.y = None

    def transform(self, X):
        self.X = X
        return X

    def fit(self, X, y=None, **fitparams):
        return self
