import pandas as pd
from gensim.models import FastText
from nltk.cluster import KMeansClusterer
import nltk
import pickle as pkl
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import cognitive_package.model.transformer as transformer_module
import cognitive_package.model.vectorizer as vectorizer_module
import os


def python_lists_to_array(k):
    list_of_arrays = list(map(lambda x: x * np.ones(M), range(k)))
    arr = np.array(list_of_arrays)
    return arr


def main():
    wordVectorFilePath = "cognitive_package/res/wordvectors/FastText/ft.txt"
    fastTextModel = FastText.load(wordVectorFilePath)
    model_type = vectorizer_module.WordVectorWrapper.GENSIM

    transformer = transformer_module.Transform2WordVectors(
        wvObject=vectorizer_module.WordVectorWrapper(fastTextModel, model_type)
    )
    root_dir = "./cognitive_package/res/pickles/"
    with open(os.path.join(root_dir, "vectorizer.pkl"), 'rb') as myfile:
        vectorizer = pkl.load(myfile)

    dataframe = pd.read_csv("cognitive_package/res/reports/keyphrases.csv")

    NUM_CLUSTERS = 3

    clusterer = KMeansClusterer(
            NUM_CLUSTERS, nltk.cluster.util.cosine_distance, avoid_empty_clusters=True)

    types = set(dataframe["type"])
    classes = set(dataframe["class"])
    print("start...")
    print(types, classes)
    for t in types:
        for c in classes:

            print("TYPE:{} ###### CLASS: {} ########".format(t, c))
            scores = dataframe.loc[(dataframe["type"] == t) & (
                dataframe["class"] == c)]["score"].values
            key_phrases = dataframe.loc[(dataframe["type"] == t) & (
                dataframe["class"] == c)]["key phrase"].values
            X = transformer.transform(vectorizer.transform(key_phrases))
            assigned_clusters = clusterer.cluster(X, assign_clusters=True)
            print(assigned_clusters)
            new_df = pd.DataFrame()
            with open('cognitive_package/res/reports/kmeans_report_{}_{}.txt'.format(t, c), 'w') as myfile:
                for k, s, a in zip(key_phrases, scores, assigned_clusters):
                    myfile.write(
                        "key_phrase: {:60s} score: {:4f} cluster: {}\n".format(k, s, a))


main()
