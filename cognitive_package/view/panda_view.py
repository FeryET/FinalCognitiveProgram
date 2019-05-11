from PyQt5 import QtCore
import pandas as pd

from PyQt5.QtWidgets import QMainWindow, QTableView


class PandaView(QMainWindow):
    def __init__(self, pd_model, parent=None):
        super().__init__(parent)
        self.width = 600
        self.height = 600
        self.table = QTableView(self)
        self.table.setModel(pd_model)
        self.setCentralWidget(self.table)

