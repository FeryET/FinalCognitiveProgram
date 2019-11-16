from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QSizePolicy,
    QHeaderView
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5 import FigureCanvasQT, FigureManagerQT
import matplotlib.pyplot as plt
import os
import cognitive_package.view.panda_view as panda_view
import cognitive_package.model.pandas_model as panda_model
import cognitive_package.view.CustomFigureCanvas as CustomFigureCanvas


class MainWindow(QMainWindow):
    on_click_browse_button_signal = pyqtSignal(str)
    on_click_start_button_signal = pyqtSignal(str)

    PLOT_TYPE_ADD_DATAPOINT = "add datapoint"
    PLOT_TYPE_PLOT_WITH_LABEL = "plot with labels"

    def __init__(self):
        super().__init__()

        self.minimumWidth = 800
        self.minimumHeight = 600
        self.maximumHeight = 1600
        self.maximumWidth = 1200

        self.__init_widgets__()
        self.__init_layout__()
        self.__connect_buttons()

    def __init_widgets__(self):
        self.central_Widget = QtWidgets.QWidget()
        self.text_box = TextInputBox(self.central_Widget)
        self.stat_table = StatTableView(self.central_Widget)
        self.start_action_button = QtWidgets.QPushButton(
            "Start Classification", self.central_Widget
        )
        self.browse_button = QtWidgets.QPushButton(
            "Browse", self.central_Widget
        )
        self.figure_canvas = CustomFigureCanvas.CustomFigureCanvas(parent=self)
        self.figure_canvas.draw()
        self.keywords_table = KeywordTableView(self)
        self.setCentralWidget(self.central_Widget)

    def __init_layout__(self):
        self.main_layout = QtWidgets.QGridLayout(self.central_Widget)
        self.main_layout.setColumnMinimumWidth = 20
        self.main_layout.setRowMinimumHeight = 20
        self.main_layout.setRowStretch = True
        self.main_layout.setColumnStretch = True
        
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.start_action_button)
        button_layout.addWidget(self.browse_button)
        button_layout.addStretch()
        # button_layout.setAlignment(Qt.AlignTop)

        self.main_layout.addLayout(button_layout, 0, 0, 3, 1)
        self.main_layout.addWidget(self.keywords_table, 4, 0, 7, 1)
        self.main_layout.addWidget(self.stat_table, 0, 1, 3, 7)        
        self.main_layout.addWidget(self.text_box, 0, 8, 11, 10)
        self.main_layout.addWidget(self.figure_canvas, 4, 1, 7, 7)

    def __connect_buttons(self):
        self.browse_button.clicked.connect(self.on_browse_button_clicked)
        self.start_action_button.clicked.connect(self.on_start_button_clicked)

    def on_start_button_clicked(self):
        print("on_start_button_clicked")
        self.on_click_start_button_signal.emit(self.text_box.toPlainText())

    def get_ax(self):
        return self.ax

    def show_dataframe(self, dfModel):
        self.dframe_dialog = panda_view.PandaView(dfModel, self)
        self.dframe_dialog.show()

    def on_browse_button_clicked(self):
        dialog = QFileDialog(
            self, "Select Directory of the Text Files", os.path.curdir
        )
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            path = dialog.selectedFiles()[0]
            self.on_click_browse_button_signal.emit(path)

    def update_table(self, pred_res, proba_res):
        self.stat_table.update_table(pred_res, proba_res)

    def get_figure_canvas(self):
        return self.figure_canvas

    def set_figure_canvas_clf(self, clf_2d):
        self.figure_canvas.set_clf_2d(clf_2d)

    def plot(self, x2D, y=None, plot_type="add_datapoint"):
        if (
            plot_type != MainWindow.PLOT_TYPE_ADD_DATAPOINT
            and plot_type != MainWindow.PLOT_TYPE_PLOT_WITH_LABEL
        ):
            raise AttributeError

        if plot_type == MainWindow.PLOT_TYPE_ADD_DATAPOINT:
            self.figure_canvas.add_datapoint(x2D)

        elif plot_type == MainWindow.PLOT_TYPE_PLOT_WITH_LABEL:
            self.figure_canvas.plot_data(x2D, y)

    def update_keywords(self, keywords):
        pass

class TextInputBox(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimumWidth = 200
        self.minimumHeight = 300

        self.placeholderText = "Type your text here..."


class StatTableView(QTableWidget):
    CLASS_NAMES = {0: "Cognitive", 1: "Not Cognitive"}
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimumWidth = 200
        self.minimumHeight = 200
        self.maximumWidth = 300
        self.maximumHeight = 300

        for i in range(3):
            self.insertRow(i)
        for i in range(3):
            self.insertColumn(i)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(False)
        font = QtGui.QFont("Arial")
        font.setPointSize(14)
        font.setBold(True)
        self.__init_table()
        # self.setSizePolicy(
        #     QtWidgets.QSizePolicy.MinimumExpanding,
        #     QtWidgets.QSizePolicy.MinimumExpanding,
        # )
        # self.setSizeAdjustPolicy(QtWidgets.QAbstractItemView.AdjustToContents)
        # self.resizeColumnsToContents()

        self.verticalHeader().setStretchLastSection(True)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

    def __init_table(self):
        self.setItem(0, 0, StatTableItem(""))
        self.item
        self.setItem(1, 0, StatTableItem("Probability"))
        self.setItem(2, 0, StatTableItem("Prediction"))

        self.setItem(0, 0, StatTableItem(""))
        self.setItem(0, 1, StatTableItem(self.CLASS_NAMES[0]))
        self.setItem(0, 2, StatTableItem(self.CLASS_NAMES[1]))

        self.setSpan(2, 1, 1, 2)

    def update_table(self, pred, proba):
        print(proba, type(proba))
        print(pred, type(pred))
        self.setItem(1, 1, StatTableItem("{:.2f}".format(proba[0, 0])))
        self.setItem(1, 2, StatTableItem("{:.2f}".format(proba[0, 1])))
        self.setItem(2, 1, StatTableItem(self.CLASS_NAMES[pred[0]]))


class StatTableItem(QTableWidgetItem):
    def __init__(self, content):
        super().__init__(content)
        self.setTextAlignment(QtCore.Qt.AlignCenter)

class KeywordTableView(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.verticalHeader().hide()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Keyword","Score"])
        # self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def update_table(self, items):
        self.clear()
        for idx, (keyword, score) in enumerate(items):
            self.insertRow(idx)
            self.setItem(idx, 0, QTableWidgetItem(keyword))
            self.setItem(idx, 1, QTableWidgetItem("{:3f}".format(score)))
        self.resizeColumnsToContents()
        self.resizeRowsToContents()