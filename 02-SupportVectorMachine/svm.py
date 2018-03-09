import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def make_data():
    X_data, y_label = make_blobs(n_samples=10, centers=3, n_features=2,
                  random_state=0)
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

    def gradient_normal(self, w, support, label, data):
        return w - np.sum(np.dot(support*label,data), axis=0)

    def gradient_bios(self, support, label):
        return np.sum(support*label)

    def gradient_support(self, label, data, w, bios):
        return label*data*w 
    
    def fit(self, train, label):
        """Train SVM"""
        self.data = train 
        self.label = label
        self.w = np.ones(self.data.shape[1])
        self.bios = np.zeros(self.w.shape[0] - 1)
        self.support = np.ones(label.shape)/len(self.label) 

        self.w -= self.gradient_normal(self.w, self.support, 
                self.label, self.data)

        self.bios -= self.gradient_bios(self.support, self.label)

        self.support -= self.gradient_support(self.label, self.data, self.w,
                self.bios)
        print(self.support)



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
    svm.fit(x_data, y_label)
    # print(svm.estimate(train) == label[:, 0])

x_data, y_label = make_data()
plot(x_data, y_label)
test_svm()
