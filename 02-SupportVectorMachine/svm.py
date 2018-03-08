import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def make_data():
    X_data, y_label = make_blobs(n_samples=50, centers=2, n_features=2,
                                 random_state=0, cluster_std=0.3)
    return X_data, y_label


def plot(data, label, train=True):
    """Plots label-colored data (2D)"""
    fig, ax = plt.subplots()
    for lab, c in zip(np.unique(label), ['red', 'green']):
        ax.scatter(
            data[:, 0][(label == lab)],
            data[:, 1][(label == lab)],
            color=c,
        )
    fig.savefig('train_svm.png')
    print('Data plotted.')


class SVM:
    """Support Vector Machine"""
    def __init__(self):
        self.w = None  # Normalenvektor
        self.b = None  # y-Achsenabschnitt (Bias)

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


# def test_svm():
#     svm = SVM()
#     svm.w = np.array([1, 0])
#     svm.b = 0
#     print(svm.estimate(train) == label[:, 0])

x_data, y_label = make_data()
plot(x_data, y_label)
# test_svm()
