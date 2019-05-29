from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    FigureManagerQT,
)
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class CustomFigureCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, cmap_name="Accent"):
        fig = Figure()
        self.color_map = plt.get_cmap(cmap_name)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setBaseSize(300, 300)
        self.setMaximumSize(400, 400)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        self.unique_labels = ["Cognitive", "Not Cognitive", "New Text"]
        self.text_colors = ["r", "b", "k"]
        self.datapoint_init_flag = True

    def set_clf_2d(self, clf_2d):
        self.clf = clf_2d

    def plot_new_datapoints(self, x2D):
        self.add_datapoint(x2D)

    @staticmethod
    def _make_meshgrid(x, y, h=0.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
        return XX, YY

    def _plot_contours(self, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        self.axes.contourf(xx, yy, Z, **params)
        self.axes.plot()
        self.axes.figure.canvas.draw()

    def plot_data(self, x2D, y):
        """plots the given array and the decision function bounday.
        
        Arguments:
            x2D {np.array} -- [2d array]
            y {np.array} -- [1d array]
        """

        x0, x1 = x2D[:, 0], x2D[:, 1]
        xx, yy = CustomFigureCanvas._make_meshgrid(x0, x1)
        labels = ["Cognitive", "Not Cognitive"]
        colors = ["r", "b"]
        self.axes.clear()
        self._plot_contours(xx, yy, cmap=self.color_map, alpha=0.8)
        target_ids = [0, 1]
        print(x2D, y)
        for i, c, label in zip(target_ids, colors, labels):
            print(i, label)
            self.axes.scatter(
                x2D[y == i, 0],
                x2D[y == i, 1],
                color=c,
                label=label,
                marker="o",
                s=(15, 15),
            )

        self.axes.set_xlim(xx.min(), xx.max())
        self.axes.set_ylim(yy.min(), yy.max())
        self.axes.set_title("2D Representation using PCA")
        self.axes.legend(fontsize=8)
        self.axes.plot()
        self.axes.figure.canvas.draw_idle()

    def add_datapoint(self, x2d):
        """Adds a new datapoint to the plot

        Arguments:
            x2d {a 2d single point, [x,y]} -- [np.array with shape (1,2)]
            axes {plt.axes} -- [description]
        
        """
        print(x2d, type(x2d))
        self.axes.scatter(
            x2d[:, 0],
            x2d[:, 1],
            color="k",
            label="Current Text",
            marker="o",
            s=(15, 15),
        )
        if self.datapoint_init_flag is True:
            self.datapoint_init_flag = False
            self.axes.legend(fontsize=8)

        self.axes.plot()
        self.axes.figure.canvas.draw()

