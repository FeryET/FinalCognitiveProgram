import matplotlib.pyplot as plt
import numpy as np


class PlotModel:
    def __init__(self, clf, cmap_name="coolwarm"):
        """to initialize a model for ploting the decision boundary
        
        Arguments:
            clf {any sklearn classifiers}
        
        Keyword Arguments:
            cmap_name {String} -- [the colormap of the meshgrid] (default: {"coolwarm"})
        """
        self.clf = clf
        self.color_map = plt.get_cmap(cmap_name)

    @staticmethod
    def _make_meshgrid(x, y, h=0.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
        return XX, YY

    def _plot_contours(self, xx, yy, axes, **params):
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
        out = axes.contourf(xx, yy, Z, **params)
        return out

    def plot_data(self, axes, x2D, y):
        """plots the given array and the decision function bounday.
        
        Arguments:
            x2D {np.array} -- [2d array]
            y {np.array} -- [1d array]
        """

        x0, x1 = x2D[:, 0], x2D[:, 1]
        xx, yy = PlotModel._make_meshgrid(x0, x1)
        labels = ["Cognitive", "Not Cognitive"]
        colors = ["r", "b"]
        axes.clear()
        self._plot_contours(xx, yy, axes, cmap=self.color_map, alpha=0.8)
        target_ids = [0, 1]
        for i, c, label in zip(target_ids, colors, labels):
            axes.scatter(
                x0[y == i, 0],
                x1[y == i, 1],
                color=c,
                label=label,
                marker="o",
                s=(15, 15),
            )

        axes.set_xlim(xx.min(), xx.max())
        axes.set_ylim(yy.min(), yy.max())
        axes.set_title("2D Representation using PCA")
        axes.legend(fontsize=8)
        return axes

    def add_datapoints(self, x2d, axes):
        """Adds a new datapoint to the plot

        Arguments:
            x2d {a 2d single point, [x,y]} -- [np.array with shape (1,2)]
            axes {plt.axes} -- [description]
        
        """
        axes.scatter(
            x2d[:, 0],
            x2d[:, 1],
            color="k",
            label="Current Text",
            marker="o",
            s=(15, 15),
        )
        plt.legend(fontsize=8)
