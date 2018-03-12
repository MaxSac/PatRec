"""pytest"""
from svm import *
import fire

n = 50
x_data, y_label = make_data(n)


def environment(epochs=1):
    """
    Baue eine Umgebung aus Testzwecken.
    Diese Umgebung sollte immer gleich aussehen, aus Gründen der Einfachkeit.
    """
    svm = SVM(epochs=epochs)
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


def func(iterations):
    """Diese Funktion wird bei `python test_svm.py` gerufen!"""
    print('Support Vector Machine (SVM):')
    print('=============================')

    svm = environment(iterations)

    # for i in range(iterations):
        # print('Iteration {}'.format(i))
    svm.estimate(x_data, y_label)

    print('Normalenvektor', svm.w)
    print('Lagrangemultiplikatoren', svm.alpha)
    print('y-Achsenabschnitt', svm.b)

    #fig, ax = Plotter().plot_hyperplane(svm.w, svm.b, x_data, y_label)
    fig, ax = Plotter().plot_label(x_data, y_label)
    fig.savefig('hyperplane.png')

    # print('Prediction:')
    # print('-----------')
    # print(svm.estimate(x_data, y_label) == y_label)
    # print('Acc:', np.mean(svm.estimate(x_data, y_label) == y_label))
    # pass


if __name__ == '__main__':
    fire.Fire(func)
