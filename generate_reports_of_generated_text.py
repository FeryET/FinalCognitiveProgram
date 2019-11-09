import os
import spacy
from spacy.lang.en import STOP_WORDS
from sklearn.tree import ExtraTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from model_evaluation import read_documents
import numpy as np
import pke
from pke.unsupervised import PositionRank
import re
import pandas as pd

mainDir = "../../datasets_of_cognitive/Data/Unprocessed Data/"
synthDir = "../../datasets_of_cognitive/Data/SynthTex/"

locs = {
    "main": "/home/farhood/Projects/datasets_of_cognitive/Data/SpellingFixed/",
    "synth":  "/home/farhood/Projects/datasets_of_cognitive/Data/SynthTexts/",
    "augmented": "/home/farhood/Projects/datasets_of_cognitive/Data/WordLevelAugmentation"
}

types = {"Cog", "NotCog"}

augmented_limits = {"synth{}".format(x) for x in range(5)}
texts = {}
for k in locs.keys():
    cur_dir = locs[k]
    for root, __, files in os.walk(cur_dir):
        if(os.path.basename(root) not in ["Cog","NotCog"]):
            continue
        type_label = "{}_{}".format(k, os.path.basename(root))
        texts[type_label] = []
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path) as myfile:
                if k == "augmented":
                    if any([x in file_name for x in augmented_limits]):
                        texts[type_label].append(myfile.read())
                else:   
                    texts[type_label].append(myfile.read())
for k in texts.keys():
    print(k, len(texts[k]))


df = pd.DataFrame(columns=["type", "class", "key phrase", "score"])

for k in texts.keys():
    print("current group", k)
    path = 'cognitive_package/res/dummy.txt'
    with open(path, 'w') as myfile:
        myfile.write("\n\n".join(texts[k]))
    with open(path) as myfile:
        print("Length of {}".format(k), len(myfile.read()))
    exctractor = PositionRank()
    exctractor.load_document(
        path, language='en', normalization='lemmatization')
    exctractor.candidate_selection(maximum_word_number=5)
    exctractor.candidate_weighting(window=30)
    text_type = k.split("_")[0]
    text_class = k.split("_")[1]

    for (keyphrase, score) in exctractor.get_n_best(n=100, redundancy_removal=True, stemming=True):
        row = pd.Series(data={"type": text_type, "class": text_class,
                        "key phrase": keyphrase, "score": score})
        df = df.append(row, ignore_index=True)

df.to_csv("cognitive_package/res/reports/keyphrases.csv")