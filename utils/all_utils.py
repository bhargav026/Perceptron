import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import ListedColormap
import logging

def prepare_data(df, target_col):
    logging.info("preparing data for training")
    X = df.drop(target_col, axis=1)

    y = df[target_col]

    return X, y

def save_plot(df, model, filename, plot_dir='plots'):
    def _create_base_plot(df):
        logging.info("crarting the base plot")
        df.plot(kind="scatter", x="X1", y='X2', c='y', s=100, cmap="coolwarm")
        plt.axhline(y=0, color='black', linestyle="--", linewidth=2)
        plt.axvline(x=0, color='black', linestyle="--", linewidth=2)

        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _plot_decision_region(X, y, classifier, resolution=0.02):
        logging.info("plotting the decison boundaries")
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        X = X.values
        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx1.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df, "y")
    _create_base_plot(df)
    _plot_decision_region(X, y, model)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)