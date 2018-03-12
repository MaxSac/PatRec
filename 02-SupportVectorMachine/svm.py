import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from tqdm import tqdm


def make_data(n):
    """Make random data"""
    x_data, y_label = make_blobs(
        n_samples=n,
        centers=2, n_features=2,
        random_state=0, cluster_std=0.7,
    )
    return x_data, y_label


class Plotter:
    def __init__(self):
        pass

    def plot_label(self, data, label):
        """Plots label-colored data (2D)"""
        fig, ax = plt.subplots()
        for lab, c in zip(np.unique(label), ['red', 'green']):
            ax.scatter(
                data[:, 0][label == lab],
                data[:, 1][label == lab],
                color=c,
            )
        print('Data plotted.')
        return fig, ax

    def plot_hyperplane(self, w, b, data, label):
        fig, ax = self.plot_label(data, label)
        x = np.linspace(0, 3, len(label))
        # ax.plot(x, (- w[0] * x + b) / w[1], label='"Normal"')
        ax.plot(x, (- b - w[0] * x) / w[1])
        # fig.legend()
        return fig, ax

    def plot_supportvectors(self, fig, ax, data, alpha):
        ax.scatter(
            data[:, 0][alpha != 0], data[:, 1][alpha != 0],
            marker='o', c='k', s=5,
        )
        return fig, ax


class SVM:
    """Support Vector Machine"""
    def __init__(self, learning_rate=0.1, epochs=1):
        self.w = None  # Normalenvektor
        self.b = None  # y-Achsenabschnitt (Bias)
        self.alpha = None  # Lagrangemultiplikatoren
        self.learning_rate = learning_rate
        self.epochs = epochs

    def gradient_weight(self, data, label):
        return self.w - np.dot((self.alpha * label), data)

    def gradient_bias(self, data, label):
        return -np.sum(self.alpha * label)

    def gradient_alpha(self, data, label):
        return label*(np.dot(data, self.w) + self.b) -1

    def encode_labels(self, label):
        u_label = np.unique(label)
        label[label == u_label[0]] = -1
        label[label == u_label[1]] = 1
        return label

    def estimate(self, train, label):
        label = self.encode_labels(label)

        self.w = np.ones(train.shape[1])
        self.alpha = np.ones(train.shape[0])
        self.alpha /= np.sum(self.alpha)
        self.b = np.ones(train.shape[1] - 1)

        for i in tqdm(range(self.epochs)):
            self.fit(train, label)

    def fit(self, train, label):
        self.w -= self.learning_rate * self.gradient_weight(train, label)
        self.w /= np.linalg.norm(self.w)
        self.b -= self.learning_rate * self.gradient_bias(train, label)
        self.alpha -= self.learning_rate * self.gradient_alpha(train, label)
        self.alpha[self.alpha < 1/train.shape[0]] = 0
        self.alpha /= np.sum(self.alpha)
        # print(self.gradient_alpha(train, label))

    def classify(self, data):
        """Classify data.
        Liegt der Punkt links oder rechts von der Trennebene?
        $sign(x \cdot w + b)$
        """
        classification = np.sign(np.dot(data, self.w) + self.b)
        return classification
