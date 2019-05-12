from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class Transform2WordVectors(BaseEstimator, TransformerMixin):
    wvObject = None

    def __init__(self, wvObject=None):
        Transform2WordVectors.wvObject = wvObject

    def fit(self, *args):
        return self

    def transform(self, *args):
        sparseX = args[0]["sparseX"]

        if not Transform2WordVectors.wvObject:  # No transformation
            return sparseX
        else:
            vocab = args[0]["vocab"]
            sortedWords = sorted(vocab, key=vocab.get)
            wordVectors = Transform2WordVectors.wvObject.getWordVectors(
                sortedWords
            )
            # sparseWordVectors = scipy.sparse.
            # reducedMatrix = sparseX * wordVectors
            reducedMatrix = self.sparseMultiply(sparseX, wordVectors)
        return reducedMatrix

    def sparseMultiply(self, sparseX, wordVectors):
        wvLength = len(wordVectors[0])
        reducedMatrix = []
        for row in sparseX:
            newRow = np.zeros(wvLength)
            for nonzeroLocation, value in list(zip(row.indices, row.data)):
                newRow = newRow + value * wordVectors[nonzeroLocation]
            reducedMatrix.append(newRow)
        reducedMatrix = np.array([np.array(x) for x in reducedMatrix])
        return reducedMatrix
