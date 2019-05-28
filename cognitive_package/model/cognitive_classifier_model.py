import os
import pickle

from pymagnitude import Magnitude
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sys

import cognitive_package.model.transformer as transformer
import cognitive_package.model.vectorizer as vectorizer
import cognitive_package.model.pipeline_placeholder as pipeline_placeholder

from gensim.models import FastText


class CognitiveClassifierModel:
    MAGNITUDE = "magnitude"
    GENSIM = "gensim"

    def __init__(self, wordvector_path, magnitude_or_gensim):
        if (
            magnitude_or_gensim != self.MAGNITUDE
            and magnitude_or_gensim != self.GENSIM
        ):
            raise AttributeError
        """This is a class which is designed to handle the binary cognitive classification task.
        Arguments:
            magModel {Type is from pymagnitude.Magnitude} -- a magnitude word2vec model 
        """

        self.w2v_model = self.__load_wordvector_model__(
            magnitude_or_gensim, wordvector_path
        )
        self.w2v_wrapper = vectorizer.WordVectorWrapper(
            self.w2v_model, magnitude_or_gensim
        )

        self.vectorizer = (
            "vectorizer",
            vectorizer.VectorizerWrapper(TfidfVectorizer()),
        )
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
        self.init_pipelines()

    def init_pipelines(self):
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

    def __load_wordvector_model__(self, magnitude_or_gensim, wordvector_path):
        if magnitude_or_gensim == self.MAGNITUDE:
            return Magnitude(wordvector_path)
        else:
            return FastText.load(wordvector_path)

    @staticmethod
    def load_pretrained(models_dir, wordvector_dir, magnitude_or_gensim):
        """This method will create a pretrained instance of CognitiveClassiferModel class.

        Arguments:
            models_dir {String} -- [where the transformer, classifier, and vectorizer models are stored. All should be in the same directory]
            mag_loc {String} -- [Where the magnitude model is stored at.]
        Returns:
            [CognitiveClassiferModel] -- [the pretrained instance]
        """
        p_Model = CognitiveClassifierModel(wordvector_dir, magnitude_or_gensim)
        for fname in os.listdir(models_dir):
            with open(os.path.join(models_dir, fname), "rb") as myfile:
                attr_name, _ = os.path.splitext(fname)
                print("loading {} starting".format(attr_name))
                pkl_file = pickle.load(myfile)
                setattr(p_Model, attr_name, (fname, pkl_file))
                print("loading {} success".format(attr_name))
        p_Model.init_pipelines()
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

    def get_X2D(self, docs):
        self.predict(docs, _2D=True)
        return self.pipeline_2d.named_steps["placeholder"].X

    def get_clf_2d(self):
        return self.clf_2d[1]
