import matplotlib.pyplot as plt
import numpy as np

from common import visualization
from common.classifiers import GaussianClassifier
from common.data_provider import DataProvider
from common.evaluation import ClassificationEvaluator


def aufg02():
    # In dieser Aufgabe soll ein Bayes'scher Normalverteilungs-Klassifikator
    # mit drei Dichten realisiert werden.
    #
    # Zunaechst sollen Mittelwert und Kovarianzmatrix der drei Klassen geschaetzt
    # und visualisiert werden:
    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    train_data, train_labels = train_data_provider.get_dataset_and_labels()

    #
    # Extrahieren Sie die Klassen-Labels aus dem Trainingsdatensatz und speichern
    # Sie sie in der lokalen Variablen labels
    #
    # Nuetzliche Funktionen:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

    raise NotImplementedError('Implement me')

    mean_list = []
    cov_list = []
    for label in labels:
        #
        # Berechnen Sie Mittelwert und Kovarianz der drei Klassen durch
        # Matrixoperationen in NumPy.
        # Speichern Sie fuer jeden Schleifendurchlauf den Mittelwert in der
        # lokalen Variablen mean und die Kovarianzmatrix in der lokalen Variablen
        # cov. Benutzen Sie zur Schaetzung die korrigierte Kovarianzmatrix:
        # https://de.wikipedia.org/wiki/Stichprobenkovarianz#Korrigierte_Stichprobenkovarianz

        raise NotImplementedError('Implement me')
        np.testing.assert_almost_equal(actual=mean,
                                       desired=np.mean(class_data, axis=0),
                                       err_msg='Der Mittelwert ist falsch')
        np.testing.assert_almost_equal(actual=cov,
                                       desired=np.cov(class_data, rowvar=0),
                                       err_msg='Die Kovarianzmatrix ist falsch')
        mean_list.append(mean)
        cov_list.append(cov)

    #
    # Visualisieren Sie die Datenpunkte der drei Klassen, sowie die geschaetzen
    # Mittelwerte und Kovarianzmatrizen durch eine Normalverteilung.
    # Zur Visualisierung der Normalverteilungen: visualization.plot_norm_dist_ellipse

    raise NotImplementedError('Implement me')

    #
    # Implementieren sie einen Bayes'schen Normalverteilungs-Klassifikator (ohne
    # Rueckweisung), der die soeben berechneten Verteilungen als Modell
    # verwendet.  Vervollstaendigen Sie dazu die Klasse GaussianClassifier im Modul
    # classification.
    #
    # Hinweise:
    #
    # Achten Sie darauf, dass Ihre Implementierung unabhaengig von den
    # verwendeten Klassenlabels ist. Aus den Trainingsdaten lassen sich diese mit
    # np.unique bestimmen.
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    #
    # Durch welche geeignete monotone Transformation lassen sich numerische
    # Probleme bei der Auswertung von extrem kleinen Dichtewerten vermeiden?
    # Beruecksichtigen Sie das in Ihrer Implementierung.

    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()
    bayes = GaussianClassifier()
    bayes.estimate(train_data, train_labels)
    estimated_labels = bayes.classify(test_data)

    #
    # Fuehren Sie eine Evaluierung der Ergebnisse wie in Aufgabe 1 durch.
    raise NotImplementedError('Implement me')

    # Ist der erstellte Klassifikator fuer diese Daten geeignet? Vergleichen Sie
    # die Ergebnisse mit dem (k)-NN-Klassifikator.

    # Diskutieren Sie Moeglichkeiten fuer eine automatische Rueckweisung.


if __name__ == '__main__':
    aufg02()
