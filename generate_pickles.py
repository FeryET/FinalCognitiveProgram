import pickle as pkl

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

import cognitive_package.model.transformer as transformer_module
import cognitive_package.model.vectorizer as vectorizer_module
import os
import re
import random
import numpy as np
import spacy
from sklearn.svm import SVC

nlp = spacy.load("en")


def read_documents(path):
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
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) in ["Cog", "NotCog"]:
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
                    label = 0 if os.path.basename(root) == "Cog" else 1
                    if len(text) < LEN_THRESHOLD:
                        print(f)
                        continue
                    dataset.append((text, label))
    random.shuffle(dataset)
    texts, labels = zip(*dataset)
    return texts, labels


# def augmentText(docs, labels):
#     for d, l in zip(docs, labels):
#         aug_sents = []
#         for sent in d.split("\n"):
#             # print(sent)
#             aug_sents.append(eda.eda(sent))
#         print(len(aug_sents))
#         for res in product(*aug_sents):
#             yield "\n".join(res), l


# def get_wordEmbeddings(doc, model):
#     result = []
#     word_list = doc.split()
#     # print(model.query(word_list))
#     for word in word_list:
#         result.append(model.query(word))
#     result = np.array(result)


def main():
    mainDir = "../../datasets_of_cognitive/Data/Unprocessed Data/"
    synthDir = "../../datasets_of_cognitive/Data/SynthTex/"

    magFilePath = (
        "cognitive_package/res/WordVectors/wiki-news-300d-1m-subword.magnitude"
    )
    magFastText = Magnitude(magFilePath)
    print("{} number of words".format(len(magFastText)))

    vec_model = TfidfVectorizer()
    vectorizer = vectorizer_module.VectorizerWrapper(model=vec_model)
    transformer = transformer_module.Transform2WordVectors(
        wvObject=vectorizer_module.WordVectorWrapper(magFastText)
    )
    pca = PCA(n_components=2)
    clf = SVC(
        kernel="linear",
        gamma="scale",
        class_weight="balanced",
        probability=True,
    )
    clf_2d = SVC(
        kernel="linear",
        gamma="scale",
        class_weight="balanced",
        probability=True,
    )
    texts, labels = read_documents(mainDir)
    texts, labels = np.array(texts), np.array(labels)
    synth_texts, synth_labels = read_documents(synthDir)
    synth_texts, synth_labels = np.array(synth_texts), np.array(synth_labels)

    count = 0
    print("training starts...")

    # train_texts, train_labels = texts[train_idx], labels[train_idx]
    # test_texts, test_labels = texts[test_idx], labels[test_idx]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2
    )
    train_texts = np.array(train_texts.tolist() + synth_texts.tolist())
    train_labels = np.array(train_labels.tolist() + synth_labels.tolist())

    vectorizer, transformer, pca, clf, clf_2d = train(
        train_texts, train_labels, vectorizer, transformer, pca, clf, clf_2d
    )
    test(test_texts, test_labels, vectorizer, transformer, pca, clf, clf_2d)
    root_dir = "./cognitive_package/res/pickles/"
    filenames = [
        (vectorizer, "vectorizer.pkl"),
        (clf, "clf.pkl"),
        (clf_2d, "clf_2d.pkl"),
        (pca, "pca.pkl"),
    ]
    for obj, f in filenames:
        with open(root_dir + f, "wb") as out_file:
            print(f)
            pkl.dump(obj, out_file, protocol=pkl.HIGHEST_PROTOCOL)


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


def test(texts, labels, vectorizer, transformer, pca, clf, clf_2d):
    X = vectorizer.transform(texts, labels)
    X_transform = transformer.transform(X, labels)
    X_2d = pca.transform(X_transform)

    print(clf.score(X_transform, labels), clf_2d.score(X_2d, labels))


if __name__ == "__main__":
    main()
