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


def plot_hyperplane(w, b, data, label):
    fig, ax = plot(data, label)
    x = np.linspace(0, 3, len(label))
    # ax.plot(x, (- w[0] * x + b) / w[1], label='"Normal"')
    ax.plot(x, w[1] * x + b, label='"Vektor"')
    fig.legend()
    return fig, ax


class SVM:
    """Support Vector Machine"""
    def __init__(self):
        self.w = None  # Normalenvektor
        self.b = None  # y-Achsenabschnitt (Bias)
        self.alpha = None  # Lagrangemultiplikatoren

    def gradient_normal(self, w, support, label, data):
        return w - np.sum(np.dot(support*label,data), axis=0)

    def gradient_bios(self, support, label):
        return np.sum(support*label)

    def gradient_support(self, label, data, w, bios):
        return label*data*w 
    
    def fit(self, train, label):
        """Train SVM"""
        learning_rate = 0.9

        # Normalenvektor schätzen und normieren
        new_w = self.w - learning_rate * (self.w - np.dot(label * self.alpha, train))
        new_w = np.linalg.norm(new_w)

        # Lagrangemultiplikatoren schätzen und normieren
        new_alpha = self.alpha - learning_rate * label * (np.dot(self.w, np.transpose(train)) + self.b - 1)
        new_alpha = new_alpha / np.sum(new_alpha)

        # y-Achsenabschnitt schätzen und normieren
        new_b = self.b - - learning_rate * np.dot(self.alpha, label)

        self.w = new_w
        self.alpha = new_alpha
        self.b = new_b

    def estimate(self, x):
        """Classify data.
        Liegt der Punkt links oder rechts von der Trennebene?
        $sign(x \cdot w + b)$
        """
        classification = np.sign(np.dot(self.w, self.w) + self.b)
        return classification


def test_svm():
    svm = SVM()
    svm.w = np.array([1, 0])
    svm.b = 0
    svm.fit(x_data, y_label)
    # print(svm.estimate(train) == label[:, 0])

def main():
    x_data, y_label = make_data(50)
    fig, ax = plot(x_data, y_label)
    # fig.savefig('train_svm.png')

x_data, y_label = make_data()
plot(x_data, y_label)
test_svm()

if __name__ == '__main__':
    main()
