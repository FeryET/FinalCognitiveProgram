from PyQt5 import QtCore
import pandas as pd
import os
from PyQt5.QtWidgets import QMainWindow, QTableView, QWidget, QHBoxLayout, QPushButton, QFileDialog, QDialog


class PandaView(QMainWindow):
    def __init__(self, pd_model, parent=None):
        super().__init__(parent)
        self.width = 600
        self.height = 600
        self.table = QTableView(self)
        self.table.setModel(pd_model)
        self.pd_model = pd_model
        widget = QWidget(self)
        hbox = QHBoxLayout()
        hbox.addWidget(self.table,75)
        hbox.addStretch(1)

        self.pushbutton = QPushButton("Save to CSV", self)
        self.pushbutton.width = 100
        self.pushbutton.height = 40
        hbox.addWidget(self.pushbutton,25)
        hbox.addStretch(1)

        widget.setLayout(hbox)

        self.setCentralWidget(widget)

        self.pushbutton.clicked.connect(self.save_to_csv_button)
    def save_to_csv_button(self):
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        path, _ = dialog.getSaveFileName(
            self, "Exporter vers", ".csv",
            "CSV Files (*.csv)")
        self.pd_model.to_csv(path)
