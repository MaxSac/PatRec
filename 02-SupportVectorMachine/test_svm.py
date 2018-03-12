"""pytest"""
from svm import *
import fire

n = 100
x_data, y_label = make_data(n)


def environment(epochs=1, learning_rate=0.1):
    """
    Baue eine Umgebung aus Testzwecken.
    Diese Umgebung sollte immer gleich aussehen, aus Gründen der Einfachkeit.
    """
    svm = SVM(epochs=epochs, learning_rate=learning_rate)
    return svm


def test_normalvector_shape():
    """Teste, ob Normalenvektor ein 2D-Vektor ist."""
    svm = environment()
    # Initialize Weights and Bias
    svm.w = np.ones(x_data.shape[1])
    svm.alpha = np.ones(x_data.shape[0])
    gradient_weight = svm.gradient_weight(x_data, y_label)
    assert gradient_weight.shape == (x_data.shape[1], )

def test_train_data():
    x_data, y_label = make_data(30)
    assert np.sum(y_label == 1) == np.sum(y_label == 0)


# def test_bias_type():
#     """Teste, ob der Achsenabschnitt ein Skalar ist."""
#     svm = environment()
#     assert type(svm.b) == int


# def test_estimation_dimension():
#     """Teste, ob die geschätzten Label die richtige Dimension haben."""
#     svm = environment()
#     assert svm.estimate(x_data).shape == y_label.shape


# def test_prediction():
#     """Teste, ob die SVM eine Genauigkeit von `acc` übersteigt."""
#     acc = 0
#     svm = environment()
#     svm.estimate(x_data, y_label)
#     assert np.mean(svm.estimate(x_data) == y_label) > acc


def func(iterations, learning_rate):
    """Diese Funktion wird bei `python test_svm.py` gerufen!"""
    print('Support Vector Machine (SVM):')
    print('=============================')

    svm = environment(iterations, learning_rate)

    # for i in range(iterations):
        # print('Iteration {}'.format(i))
    svm.estimate(x_data, y_label)

    # print('Normalenvektor', svm.w)
    # print('Lagrangemultiplikatoren', svm.alpha)
    # print('y-Achsenabschnitt', svm.b)

    plotter = Plotter()
    fig, ax = plotter.plot_hyperplane(svm.w, svm.b, x_data, y_label)
    plotter.plot_supportvectors(fig, ax, x_data, svm.alpha)
    fig.savefig('hyperplane.png')

    print('Prediction:')
    print('-----------')
    print(svm.classify(x_data) == y_label)
    print('Acc:', np.mean(svm.classify(x_data) == y_label))


if __name__ == '__main__':
    fire.Fire(func)
