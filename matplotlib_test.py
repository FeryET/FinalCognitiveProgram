import sys
from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from cognitive_package.view.CustomFigureCanvas import CustomFigureCanvas

import random
import numpy as np
from sklearn.svm import SVC


class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = CustomFigureCanvas(parent=self)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QtWidgets.QPushButton("Plot")
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self):

        x2D = np.random.rand(50, 2)
        y = np.random.randint(0, 2, size=(50,))

        x2D_train = np.random.rand(50, 2)
        y_train = np.random.randint(0, 2, size=(50,))

        clf = SVC()
        clf.fit(x2D_train, y_train)

        print(x2D)
        self.canvas.set_clf_2d(clf)
        self.canvas.plot_data(x2D, y)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
