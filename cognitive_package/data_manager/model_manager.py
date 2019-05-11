import model.cognitive_classifier_model as cognitive_classifier_model
import model.plot_model as plot_model
import os
import util.text_cleaner as text_cleaner


class ModelManager:
    def __init__(self):
        super().__init__()
        self.model = None
        self.plot_util = None
        self.cleaner = text_cleaner.TextCleaner()

    def initialize_model_pretrained(self, mag_loc, models_loc):
        self.model = cognitive_classifier_model.CognitiveClassifierModel.load_pretrained(
            models_loc, mag_loc
        )
        self.plot_util = plot_model.PlotModel(self.model.get_clf_2d())

    def initialize_model(self, mag_loc):
        self.model = cognitive_classifier_model.CognitiveClassifierModel(
            mag_loc
        )
        self.plot_util = plot_model.PlotModel(self.model.get_clf_2d())

    def predict(self, texts):
        pred = self.model.predict(texts)
        pred_proba = self.model.predict_proba(texts)
        return pred, pred_proba

    def clean_text(self, text):
        return self.clean_text(text)

    def read_texts_by_subdirs(self, path, class_names=["Cog", "NotCog"]):
        texts = []
        labels = []
        classes = {x: id for id, x in enumerate(class_names)}
        fnames = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".txt"):
                    with open(os.path.join(root, f), "r") as myfile:
                        texts.append(self.clean_text(myfile.read()))
                        labels.append(classes[os.path.basename(root)])
                        fnames.append(f)
        return texts, labels, fnames

    def read_texts_in_folder(self, path):
        texts = []
        fnames = []
        for root, _, files in os.walk(path):
            for f in files:
                with open(os.path.join(root, f), "r") as readFile:
                    texts.append(self.clean_text(readFile.read()))
                    fnames.append(f)
        return texts, fnames

    def plot_data(self, ax, x2D, y):
        return self.plot_util.plot_data(ax, x2D, y)

    def pre_visualize_data(self, ax, texts, labels):
        x2D = self.model.get_X2D(texts)
        return self.plot_data(ax, x2D, labels)

    def plot_new_datapoints(self, ax, texts):
        x2d = self.model.get_X2D(texts)
        return self.plot_util.add_datapoint(x2d, ax)
