import os
import pickle

from pymagnitude import Magnitude
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sys

import transformer
import vectorizer
import pipeline_placeholder


class CognitiveClassifierModel:
    def __init__(self, mag_loc):
        """This is a class which is designed to handle the binary cognitive classification task.
        Arguments:
            magModel {Type is from pymagnitude.Magnitude} -- a magnitude word2vec model 
        """
        self.vectorizer = (
            "vectorizer",
            vectorizer.VectorizerWrapper(TfidfVectorizer()),
        )
        self.w2v_model = Magnitude(mag_loc)
        self.w2v_wrapper = vectorizer.WordVectorWrapper(self.w2v_model)
        self.transformer = (
            "transformer",
            transformer.Transform2WordVectors(wvObject=self.w2v_wrapper),
        )
        self.clf = (
            "classifier",
            SVC(
                kernel="linear",
                gamma="scale",
                class_weight="balanced",
                probability=True,
            ),
        )
        self.clf_2d = (
            "classifier_2d",
            SVC(
                kernel="linear",
                gamma="scale",
                class_weight="balanced",
                probability=True,
            ),
        )
        self.pca = ("pca", PCA(n_components=2))
        self.placeholder = (
            "placeholder",
            pipeline_placeholder.PipelinePlaceHolder(),
        )
        self.pipeline = Pipeline([self.vectorizer, self.transformer, self.clf])
        self.pipeline_2d = Pipeline(
            [
                self.vectorizer,
                self.transformer,
                self.pca,
                self.placeholder,
                self.clf_2d,
            ]
        )

    @staticmethod
    def load_pretrained(models_dir, mag_loc):
        """This method will create a pretrained instance of CognitiveClassiferModel class.

        Arguments:
            models_dir {String} -- [where the transformer, classifier, and vectorizer models are stored. All should be in the same directory]
            mag_loc {String} -- [Where the magnitude model is stored at.]
        Returns:
            [CognitiveClassiferModel] -- [the pretrained instance]
        """
        p_Model = CognitiveClassifierModel(mag_loc)
        for fname in os.listdir(models_dir):
            with open(os.path.join(models_dir, fname), "rb") as myfile:
                attr_name, _ = os.path.splitext(fname)
                setattr(p_Model, attr_name, pickle.load(myfile))
        return p_Model

    def fit(self, docs, labels, _2D=False):
        if _2D is False:
            self.pipeline.fit(docs, labels)
        else:
            self.pipeline_2d.fit(docs, labels)

    def score(self, docs, labels, _2D=False):
        if _2D is False:
            return self.pipeline.score(docs, labels)
        else:
            return self.pipeline_2d.score(docs, labels)

    def predict_proba(self, docs, _2D=False):
        if _2D is False:
            return self.pipeline.predict_proba(docs)
        else:
            return self.pipeline_2d.predict_proba(docs)

    def predict(self, docs, _2D=False):
        if _2D is False:
            return self.pipeline.predict(docs)
        else:
            return self.pipeline_2d.predict(docs)

    def get_X2D(self, X):
        return self.pipeline_2d.named_steps["placeholder"].X

    def get_clf_2d(self):
        return self.clf_2d
