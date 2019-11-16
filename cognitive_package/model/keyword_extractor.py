from pke.unsupervised import PositionRank


class KeyWordExtractor:
    def __init__(self):
        self.__extractor = PositionRank()

    def get_keywords(self, text, num_of_keywords=10):
        self.__extractor.load_document(
            text, language='en', normalization='lemmatization')
        self.__extractor.candidate_selection(maximum_word_number=3)
        self.__extractor.candidate_weighting(window=10)
        result = []
        for keyphrase, score in self.__extractor.get_n_best(n=num_of_keywords,
                                                redundancy_removal=False,
                                                stemming=False):
            result.append((keyphrase, score))
