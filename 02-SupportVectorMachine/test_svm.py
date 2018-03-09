"""pytest"""
from svm import *
import fire

n = 50
x_data, y_label = make_data(n)


def environment():
    """
    Baue eine Umgebung aus Testzwecken.
    Diese Umgebung sollte immer gleich aussehen, aus Gründen der Einfachkeit.
    """
    svm = SVM()

    svm.w = np.array([1, 0])
    svm.b = 0

    return svm


def test_normalvector_shape():
    """Teste, ob Normalenvektor ein 2D-Vektor ist."""
    svm = environment()
    assert svm.w.shape == (2,)


def test_bias_type():
    """Teste, ob der Achsenabschnitt ein Skalar ist."""
    svm = environment()
    assert type(svm.b) == int


def test_estimation_dimension():
    """Teste, ob die geschätzten Label die richtige Dimension haben."""
    svm = environment()
    assert svm.estimate(x_data).shape == y_label.shape


def test_prediction():
    """Teste, ob die SVM eine Genauigkeit von `acc` übersteigt."""
    acc = 0
    svm = environment()
    svm.fit(x_data, y_label)
    assert np.mean(svm.estimate(x_data) == y_label) > acc


def test(iterations):
    """Diese Funktion wird bei `python test_svm.py` gerufen!"""
    print('Support Vector Machine (SVM):')
    print('=============================')

    svm = environment()
    svm.w = np.array([1, 1])
    svm.b = 1
    svm.alpha = np.ones(len(x_data))

    for i in range(iterations):
        # print('Iteration {}'.format(i))
        svm.fit(x_data, y_label)

        # print('Normalenvektor', svm.w)
        # print('Lagrangemultiplikatoren', svm.alpha)
        # print('y-Achsenabschnitt', svm.b)

    print(svm.b)

    fig, ax = plot_hyperplane(svm.w, svm.b, x_data, y_label)
    fig.savefig('hyperplane.png')

    print('Prediction:')
    print('-----------')
    print(svm.estimate(x_data) == y_label)
    print('Acc:', np.mean(svm.estimate(x_data) == y_label))
    pass


if __name__ == '__main__':
    fire.Fire(test)
