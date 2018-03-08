"""pytest"""
from svm import *

n = 10
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


def test():
    """Diese Funktion wird bei `python test_svm.py` gerufen!"""
    print('Support Vector Machine (SVM):')
    print('-----------------------------')

    svm = environment()
    svm.fit(x_data, y_label)

    print('Prediction:')
    print(svm.estimate(x_data) == y_label)
    # print(np.mean(svm.estimate(x_data) == y_label))
    # print('Alpha', svm.alpha.shape)
    # print(svm.alpha)
    # print('Normalenvironmentektor', svm.w.shape)
    # print(svm.w)
    # print('Achsenabschnitt')
    # print(svm.b)
    pass


if __name__ == '__main__':
    test()
