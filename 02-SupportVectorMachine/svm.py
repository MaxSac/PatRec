import numpy as np
import matplotlib.pyplot as plt

train = np.array(
    [
        [-1, -5], [-1, -4], [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5],
        [1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
    ]
)

label = np.array(
    [
        [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
        [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
    ]
)


def plot(data, label, train=True):
    """Plots label-colored data (2D)"""
    fig, ax = plt.subplots()
    for lab, c in zip(np.unique(label), ['red', 'green']):
        ax.scatter(
            data[:, 0][(label == lab)[:, 0]],
            data[:, 1][(label == lab)[:, 0]],
            color=c,
        )
    fig.savefig('train_svm.png')
    print('Data plotted.')


# plot(train, label)


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


def test_svm():
    svm = SVM()
    svm.w = np.array([1, 0])
    svm.b = 0
    print(svm.estimate(train) == label[:, 0])


test_svm()
