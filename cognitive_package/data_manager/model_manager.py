import cognitive_package.model.cognitive_classifier_model as cognitive_classifier_model
import cognitive_package.model.plot_model as plot_model
import os
import cognitive_package.util.text_cleaner as text_cleaner


class ModelManager:
    def __init__(self, gensim_or_magnitude):
        super().__init__()
        self.model = None
        self.gensim_or_magnitude = gensim_or_magnitude
        self.cleaner = text_cleaner.TextCleaner()

    def initialize_model_pretrained(self, w2v_loc, models_loc):
        self.model = cognitive_classifier_model.CognitiveClassifierModel.load_pretrained(
            models_loc, w2v_loc, self.gensim_or_magnitude
        )

    def initialize_model(self, w2v_loc):
        self.model = cognitive_classifier_model.CognitiveClassifierModel(
            w2v_loc, self.gensim_or_magnitude
        )

    def predict(self, texts):
        pred = self.model.predict(texts)
        pred_proba = self.model.predict_proba(texts)
        return pred, pred_proba

    def clean_text(self, text):
        return self.cleaner.clean_text(text)

    def read_texts_by_subdirs(self, path, class_names=["Cog", "NotCog"]):
        texts = []
        labels = []
        classes = {x: id for id, x in enumerate(class_names)}
        fnames = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".txt"):
                    with open(os.path.join(root, f), "r") as myfile:
                        texts.append(self.cleaner.clean_text(myfile.read()))
                        labels.append(classes[os.path.basename(root)])
                        fnames.append(f)
        return texts, labels, fnames

    def read_texts_in_folder(self, path):
        texts = []
        fnames = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".txt"):
                    try:
                        with open(
                            os.path.join(root, f),
                            "r",
                            encoding="utf-8",
                            errors="strict",
                        ) as readFile:
                            texts.append(self.clean_text(readFile.read()))
                            fnames.append(f)
                    except:
                        pass
        return texts, fnames

    def get_x2D(self, texts):
        return self.model.get_X2D(texts)

    def get_clf2D(self):
        return self.model.get_clf_2d()
