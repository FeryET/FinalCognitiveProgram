import re

import spacy
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


class TextCleaner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.filter = [
            lambda x: x.lower(),
            strip_multiple_whitespaces,
            strip_numeric,
            strip_non_alphanum,
            strip_punctuation,
            remove_stopwords,
            strip_tags,
            lambda s: strip_short(s, minsize=4),
        ]

    def clean_text(self, text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        res = []
        doc = self.nlp(text)
        for sent in doc.sents:
            sent = " ".join([word.lemma_ for word in sent])
            res.append(" ".join(preprocess_string(sent, filters=self.filter)))
        text = "\n".join(res)
        return text
