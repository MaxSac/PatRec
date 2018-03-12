import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize


def make_data(n):
    """Make random data"""
    x_data, y_label = make_blobs(
        n_samples=n,
        centers=2, n_features=2,
        random_state=0, cluster_std=0.3,
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
        ax.plot(x, w[1] * x + b)
        # fig.legend()
        return fig, ax


class SVM:
    """Support Vector Machine"""
    def __init__(self, learning_rate=0.05, epochs=1):
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
        label[u_label[0]] = -1
        label[u_label[1]] = 1
        return label

    def estimate(self, train, label):
        label = self.encode_labels(label)

        self.w = np.ones(train.shape[1])
        self.alpha = np.ones(train.shape[0])
        self.alpha = self.alpha / np.linalg.norm(self.alpha)
        self.b = np.ones(train.shape[1] - 1)

        for i in range(self.epochs):
            self.fit(train, label)

    def fit(self, train, label):
        self.w -= self.learning_rate * self.gradient_weight(train, label)
        self.w /= np.linalg.norm(self.w)
        self.b -= self.learning_rate * self.gradient_bias(train, label)
        self.alpha -= self.learning_rate * self.gradient_alpha(train, label)
        print(self.gradient_alpha(train, label))

    def classify(self, data):
        """Classify data.
        Liegt der Punkt links oder rechts von der Trennebene?
        $sign(x \cdot w + b)$
        """
        classification = np.sign(np.dot(self.w, self.w) + self.b)
        return classification


# def test_svm():
#     svm = SVM()
#     svm.estimate(x_data, y_label)

def main(number_samples):
    x_data, y_label = make_data(50)
    svm = SVM()
    svm.fit(x_data, y_label)
    fig, ax = plot(x_data, y_label)
    fig.show()


if __name__ == '__main__':
    main(number_samples)
