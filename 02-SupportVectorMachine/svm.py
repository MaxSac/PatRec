import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def make_data(n):
    """Make random data"""
    x_data, y_label = make_blobs(
        n_samples=n,
        centers=2, n_features=2,
        random_state=0, cluster_std=0.3,
    )
    return x_data, y_label


def plot(data, label):
    """Plots label-colored data (2D)"""
    fig, ax = plt.subplots()
    for lab, c in zip(np.unique(label), ['red', 'green']):
        ax.scatter(
            data[:, 0][(label == lab)],
            data[:, 1][(label == lab)],
            color=c,
        )
    print('Data plotted.')
    return fig, ax


class SVM:
    """Support Vector Machine"""
    def __init__(self):
        self.w = None  # Normalenvektor
        self.b = None  # y-Achsenabschnitt (Bias)
        self.alpha = None  # Lagrangemultiplikatoren

    def fit(self, train, label):
        """Train SVM"""
        pass

    def estimate(self, x):
        """Classify data.
        Liegt der Punkt links oder rechts von der Trennebene?
        $sign(x \cdot w + b)$
        """
        classification = np.sign(np.dot(x, self.w) + self.b)
        return classification


def main():
    x_data, y_label = make_data(50)
    fig, ax = plot(x_data, y_label)
    # fig.savefig('train_svm.png')


if __name__ == '__main__':
    main()
