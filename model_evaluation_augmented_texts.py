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

FILE_NAME = 'file_name'
TEXT = 'original_text'
LABEL = 'label'
AUGMENTED_TEXTS = 'augmented_texts'
AUGMENTED_LABELS = 'augmented_labels'


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
                    label = labels_dict[os.path.basename(root)]
                    if len(text) < LEN_THRESHOLD:
                        print(f)
                        continue
                    dataset.append((f, text, label))
    random.shuffle(dataset)
    filenames, texts, labels = zip(*dataset)
    return filenames, texts, labels


def bind_original_and_augmented(
    original_filenames,
    original_texts,
    original_labels,
    augmented_filenames,
    augmented_texts,
    augmented_labels
):
    bounded = []
    for file_name, label, text in zip(original_filenames, original_labels, original_texts):
        current_cell = {
            FILE_NAME: file_name,
            LABEL: label,
            TEXT: text,
            AUGMENTED_TEXTS: [],
            AUGMENTED_LABELS: [],
        }
        for aug_file_name, aug_text, aug_label in zip(augmented_filenames, augmented_texts, augmented_labels):
            if file_name[:-4] in aug_file_name:
                current_cell[AUGMENTED_TEXTS].append(aug_text)
                current_cell[AUGMENTED_LABELS].append(aug_label)
        bounded.append(current_cell)
    return bounded

def check_over_lapping_file_names(source_path, des_path):
    for source in os.listdir(source_path):
        for other in os.listdir(des_path):
            if source != other and source[:-4] in other[:-4]:
                print(source, other)


def custom_kfold(data, n_splits=10, n_repeats=5):
    interval = int(len(data) / n_splits)
    for _ in range(n_repeats):
        indices = np.arange(len(data))
        random.shuffle(indices)
        start_idx = 0
        for __ in range(n_splits):
            stop_idx = start_idx + \
                interval if ((start_idx + interval) <
                             len(indices)) else len(indices) - 1
            test_idx = indices[start_idx: stop_idx]
            train_idx = indices[:start_idx]
            if(stop_idx < len(indices)):
                train_idx = np.concatenate((train_idx, indices[stop_idx:]))
            start_idx += interval
            yield train_idx, test_idx


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

def main():

    
    print('start...')

    mainDir = "/home/farhood/Projects/datasets_of_cognitive/Data/SpellingFixed/"
    synthDir = "/home/farhood/Projects/datasets_of_cognitive/Data/WordLevelAugmentation/"
    
    print('reading documents...')
    original_filenames, original_texts, original_labels = read_documents(
        mainDir)

    original_texts, original_labels = np.array(
        original_texts), np.array(original_labels)
    aug_filenames, aug_texts, aug_labels = read_documents(synthDir)
    aug_texts, aug_labels = np.array(aug_texts), np.array(aug_labels)

    dataset = np.array(bind_original_and_augmented(
        original_filenames, original_texts, original_labels,
        aug_filenames, aug_texts, aug_labels
    ))

    print("LENGTH OF DATASET: {}".format(len(dataset)))
    print("Shannon Entropy of dataset: {}".format(entropy(original_labels)))
    print("Values and counts of different classes in dataset: {}".format(
        np.unique(original_labels, return_counts=True)))
    print('loading word embedder model...')
    wordVectorFilePath = "cognitive_package/res/wordvectors/wiki-news-300d-1m-subword.magnitude"
    wordEmbedderModel = Magnitude(wordVectorFilePath)
    model_type = vectorizer_module.WordVectorWrapper.MAGNITUDE
    print("{} number of words".format(len(wordEmbedderModel)))

    # wordVectorFilePath = "cognitive_package/res/wordvectors/FastText/ft.txt"
    # wordEmbedderModel = FastText.load(wordVectorFilePath)
    # model_type = vectorizer_module.WordVectorWrapper.GENSIM

    print("training starts...")

    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    conf_mats = []
    roc_auc_scores = []
    log_losses = []
    f1_scores = []
    rmse_scores = []
    for step, (train_idx, test_idx) in enumerate(custom_kfold(dataset, n_splits=10, n_repeats=5)):

        train_texts, train_labels = [], []
        test_texts, test_labels = [], []

        for item in dataset[train_idx].tolist():
            train_texts += [item[TEXT]] + item[AUGMENTED_TEXTS]
            train_labels += [item[LABEL]] + item[AUGMENTED_LABELS]

        for item in dataset[test_idx].tolist():
            test_texts.append(item[TEXT])
            test_labels.append(item[LABEL])

        vec_model = TfidfVectorizer()
        vectorizer = vectorizer_module.VectorizerWrapper(model=vec_model)
        transformer = transformer_module.Transform2WordVectors(
            wvObject=vectorizer_module.WordVectorWrapper(
                wordEmbedderModel, model_type)
        )
        clf = SVC(
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
        )

        pipeline = Pipeline(
            steps=[('vectorizer', vectorizer),
                   ('transformer', transformer),

                   ('clf', clf)]
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
        print("current step: {}\taccuracy: {:3f}".format(step, accuracy))
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
