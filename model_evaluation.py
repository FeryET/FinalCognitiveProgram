import pickle as pkl
import pandas as pd
from gensim.parsing.preprocessing import (
    preprocess_string,
    remove_stopwords,
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_numeric,
    strip_punctuation,
    strip_short,
    strip_tags,
)
from gensim.summarization.textcleaner import split_sentences
from pymagnitude import Magnitude
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss, f1_score, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold
import cognitive_package.model.transformer as transformer_module
import cognitive_package.model.vectorizer as vectorizer_module
import os
import re
import random
import numpy as np
import spacy
from sklearn.svm import SVC
from math import log, e

from gensim.models import FastText

import logging


nlp = spacy.load("en_core_web_sm")


def read_documents(path):
    labels_dict = {"Cog": 0, "NotCog": 1}
    dataset = []
    filter = [
        lambda x: x.lower(),
        strip_multiple_whitespaces,
        strip_numeric,
        strip_non_alphanum,
        strip_punctuation,
        remove_stopwords,
        strip_tags,
        lambda s: strip_short(s, minsize=4),
    ]
    LEN_THRESHOLD = 10
    for root, _, files in os.walk(path):
        if os.path.basename(root) in labels_dict.keys():
            print(root)
            for f in files:
                with open(os.path.join(root, f), "r") as myfile:
                    text = myfile.read()
                    text = re.sub(r"[^\x00-\x7F]+", " ", text)
                    res = []
                    doc = nlp(text)
                    for sent in doc.sents:
                        sent = " ".join([word.lemma_ for word in sent])
                        res.append(
                            " ".join(preprocess_string(sent, filters=filter))
                        )
                    text = "\n".join(res)
                    label = labels_dict[root]
                    if len(text) < LEN_THRESHOLD:
                        print(f)
                        continue
                    dataset.append((text, label))
    random.shuffle(dataset)
    texts, labels = zip(*dataset)
    return texts, labels

def preprocess_documents(path, preprocess_folder_root):
    
def train(texts, labels, vectorizer, transformer, pca, clf, clf_2d):
    X = vectorizer.fit_transform(texts, labels)
    print("vectorizer is trained.")
    X_transform = transformer.transform(X, labels)
    print("transformer is trained.")
    clf.fit(X_transform, labels)
    print("main classifier is trained.")

    X_2d = pca.fit_transform(X_transform)
    print("pca is trained.")
    clf_2d.fit(X_2d, labels)
    print("clf 2d is trained.")

    return vectorizer, transformer, pca, clf, clf_2d


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def test(texts, labels, vectorizer, transformer, pca, clf, clf_2d):
    X = vectorizer.transform(texts, labels)
    X_transform = transformer.transform(X, labels)
    X_2d = pca.transform(X_transform)

    print(clf.score(X_transform, labels), clf_2d.score(X_2d, labels))


def main():
    print('start...')

    mainDir = "../../datasets_of_cognitive/Data/Unprocessed Data/"
    synthDir = "../../datasets_of_cognitive/Data/2000_gpt_generated/"

    print('reading documents...')
    original_texts, original_labels = read_documents(mainDir)
    original_texts, original_labels = np.array(
        original_texts), np.array(original_labels)
    synth_texts, synth_labels = read_documents(synthDir)
    synth_texts, synth_labels = np.array(synth_texts), np.array(synth_labels)
    print("Shannon Entropy of dataset: {}".format(entropy(original_labels)))
    print("Values and counts of different classes in dataset: {}".format(
        np.unique(original_labels, return_counts=True)))
    print('loading fast text model...')
    # wordVectorFilePath = "cognitive_package/res/wordvectors/wiki-news-300d-1m-subword.magnitude"
    # fastTextModel = Magnitude(wordVectorFilePath)
    # model_type = vectorizer_module.WordVectorWrapper.MAGNITUDE
    # print("{} number of words".format(len(fastTextModel)))

    wordVectorFilePath = "cognitive_package/res/wordvectors/FastText/ft.txt"
    fastTextModel = FastText.load(wordVectorFilePath)
    model_type = vectorizer_module.WordVectorWrapper.GENSIM

    print("training starts...")

    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

    conf_mats = []
    roc_auc_scores = []
    log_losses = []
    f1_scores = []
    rmse_scores = []
    count = 0
    for train_idx, test_idx in kfold.split(original_texts, original_labels):
        print("current step: {}".format(count))
        count += 1
        train_texts = np.array(
            original_texts[train_idx].tolist() + synth_texts.tolist())
        train_labels = np.array(
            original_labels[train_idx].tolist() + synth_labels.tolist())
        test_texts = original_texts[test_idx]
        test_labels = original_labels[test_idx]

        vec_model = TfidfVectorizer()
        vectorizer = vectorizer_module.VectorizerWrapper(model=vec_model)
        transformer = transformer_module.Transform2WordVectors(
            wvObject=vectorizer_module.WordVectorWrapper(
                fastTextModel, model_type)
        )
        clf = SVC(
            kernel="linear",
            gamma="scale",
            class_weight="balanced",
            probability=True,
        )

        pipeline = Pipeline(
            steps=[('vectorizer', vectorizer),
                   ('transformer', transformer), ('clf', clf)]
        )

        pipeline.fit(train_texts, train_labels)
        predicted = pipeline.predict(test_texts)
        conf_mats.append(confusion_matrix(test_labels, predicted))
        log_losses.append(log_loss(test_labels, predicted))
        roc_auc_scores.append(roc_auc_score(test_labels, predicted))
        f1_scores.append(f1_score(test_labels, predicted))
        rmse_scores.append(np.sqrt(mean_squared_error(test_labels, predicted)))
        accuracy = ((conf_mats[-1][0][0] + conf_mats[-1]
                     [1][1]) / conf_mats[-1].sum())
        print("accuracy: {:3f}".format(accuracy))
    df = pd.DataFrame()
    df['confusion_matrice'] = conf_mats
    df['roc_auc_score'] = roc_auc_scores
    df['log_loss'] = log_losses
    df['f1_score'] = f1_scores
    df['rmse'] = rmse_scores
    df.to_pickle(
        'cognitive_package/res/reports/model evaluation/evaluation_results.pkl')


if __name__ == "__main__":
    main()
