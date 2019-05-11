import pandas as pd
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication
import data_manager.model_manager as model_manager
import view.main_window as main_window
import model.pandas_model as pandas_model


class CognitiveController(QApplication):
    def __init__(self, path, **argv):
        super().__init__(argv)
        self.dataManager = model_manager.ModelManager()
        self.view = main_window.MainWindow()
        self.init_texts_path = path
        self.BULK_TEXT_CLASSIFCATION_PATH = "result.txt"
        self.__onStartUp__()
        self.view.show()

    def __onStartUp__(self):
        texts, labels, _ = self.dataManager.read_texts_by_subdirs(
            self.init_texts_path
        )
        self.dataManager.pre_visualize_data(self.view.get_ax(), texts, labels)

    @pyqtSlot(str)
    def onPredictClickListener(self, text):
        pred, proba = self.dataManager.predict([text])
        self.view.update_table(pred, proba)
        self.dataManager.plot_new_datapoints(self.view.get_ax(), [text])

    @pyqtSlot(str)
    def onPathSelectedListener(self, path):
        texts, fnames = self.dataManager.read_texts_in_folder(path)
        pred_list, proba_list = self.dataManager.predict(texts)
        self.dataManager.plot_new_datapoints(self.view.get_ax(), [texts])
        dframe = pd.DataFrame(zip(*zip(fnames, pred_list, zip(*proba_list))))
        pd_model = pandas_model.PandasModel(dframe)
        self.view.show_dataframe()

    def __connect_signals__(self):
        self.view.on_browse_button_clicked.connect(self.onPathSelectedListener)
        self.view.on_click_start_button.connect(self.onPredictClickListener)
