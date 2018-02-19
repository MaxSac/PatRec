import numpy as np
from sklearn import svm

from common.data_provider import DataProvider
from common.evaluation import ClassificationEvaluator


def aufg08():
    #
    # Diese Aufgabe ist Optional
    #
    #
    # Die Leistungsfaehigkeit der Support Vector Maschinen soll nun fuer das
    # realistischere Szenario der Zeichenerkennung untersucht werden.

    # Erstellen Sie ein komplettes Klassifikationssystem fuer den MNIST-Datensatz
    # in Originalrepraesentation (784-D) auf der Basis einer linearen SVM
    # zunaechst mit den Standardparametern von sklearn.
    # Vergleichen Sie das Klassifikationsergebnis mit denen der vorherigen Klassifikatoren.

    raise NotImplementedError('Implement me')

    # Verwenden Sie an Stelle der 784-dimensionalen Originaldaten die
    # Merkmalsrepraesentationen der vorhergehenden Uebungsblaetter. Die jeweiligen
    # SVMs sollen auf der Basis einer linearen SVM mit soft-margin erstellt
    # werden. Vergleichen Sie die Ergebnisse.

    raise NotImplementedError('Implement me')

    # Variieren Sie die verwendete Kernelfunktion und vergleichen Sie Ihre Ergebnisse.
    raise NotImplementedError('Implement me')

    # Evaluieren Sie, wie sich Veraenderungen der jeweiligen Kernelparameter auf
    # die entsprechenden Klassifikationsergebnisse auswirken.
    raise NotImplementedError('Implement me')


if __name__ == '__main__':
    aufg08()
