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
)
from matplotlib.backends.backend_qt5 import FigureCanvasQT, FigureManagerQT
import matplotlib.pyplot as plt
import os
import panda_view


class MainWindow(QMainWindow):
    on_click_browse_button = pyqtSignal(str)
    on_click_start_button = pyqtSignal(str)

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
        self.text_box = CustomEditTextBox(self.central_Widget)
        self.stat_table = CustomTableView(self.central_Widget)
        self.start_action_button = QtWidgets.QPushButton(
            "Start Classification", self.central_Widget
        )
        self.browse_button = QtWidgets.QPushButton(
            "Browse", self.central_Widget
        )
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.figure_canvas = CustomFigureCanvas(self.fig)

        self.setCentralWidget(self.central_Widget)

    def __init_layout__(self):
        self.layout = QtWidgets.QGridLayout(self.central_Widget)
        self.layout.setColumnMinimumWidth = 20
        self.layout.setRowMinimumHeight = 20
        self.layout.setRowStretch = True
        self.layout.setColumnStretch = True
        self.layout.addWidget(self.start_action_button, 0, 0, 1, 1)
        self.layout.addWidget(self.browse_button, 1, 0, 1, 1)
        self.layout.addWidget(self.stat_table, 0, 1, 3, 3)
        self.layout.addWidget(self.text_box, 0, 8, 10, 10)
        self.layout.addWidget(self.figure_canvas, 4, 1, 7, 7)

    def __connect_buttons(self):
        self.browse_button.clicked.connect(self.on_browse_button_clicked)
        self.start_action_button.clicked.connect(self.on_browse_button_clicked)

    @pyqtSlot()
    def on_start_button_clicked(self):
        self.on_start_button_clicked.emit(self.text_box.toPlainText())

    def get_ax():
        return self.ax

    def show_dataframe(dfModel):
        self.dframe_dialog = panda_view.PandaView(pd_model, self)
        self.dframe_dialog.show()

    def on_browse_button_clicked(self):
        dialog = QFileDialog(
            self, "Select Directory of the Text Files", os.path.curdir
        )
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            path = dialog.selectedFiles()[0]
            self.on_click_browse_button.emit(path)

    def update_table(self, pred_res, proba_res):
        self.stat_table.update_table(pred, proba)


class CustomEditTextBox(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimumWidth = 200
        self.minimumHeight = 300

        self.placeholderText = "Type your text here..."


class CustomTableView(QTableWidget):
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

        self.classNames = {0: "Cognitive", 1: "Not Cognitive"}
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
        self.setItem(0, 0, CustomTableItem(""))
        self.item
        self.setItem(1, 0, CustomTableItem("Probability"))
        self.setItem(2, 0, CustomTableItem("Prediction"))

        self.setItem(0, 0, CustomTableItem(""))
        self.setItem(0, 1, CustomTableItem(self.classNames[0]))
        self.setItem(0, 2, CustomTableItem(self.classNames[1]))

        self.setSpan(2, 1, 1, 2)

    def update_table(self, pred, proba):
        self.setItem(1, 1, CustomTableItem("{:.2f}".format(proba[0])))
        self.setItem(1, 2, CustomTableItem("{:.2f}".format(proba[1])))
        self.setItem(2, 1, CustomTableItem(self.classNames[pred]))


class CustomFigureCanvas(FigureCanvasQT):
    def __init__(self, fig):
        super().__init__(fig)
        self.setBaseSize(300, 300)
        self.setMaximumSize(400, 400)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )


class CustomTableItem(QTableWidgetItem):
    def __init__(self, content):
        super().__init__(content)
        self.setTextAlignment(QtCore.Qt.AlignCenter)

