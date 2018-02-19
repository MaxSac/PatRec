import matplotlib.pyplot as plt
from sklearn import svm
from common.data_provider import DataProvider
from common.evaluation import ClassificationEvaluator
from common import visualization


def aufg07():
    #
    # Im Gegensatz zu den bisher verwendeten Klassifikatoren, die Klassengebiete
    # explizit ueber (Mischungen von) Gaussverteilungen modellieren, sollen nun
    # diskriminative Klassifikatoren untersucht werden. Als besonders leistungs-
    # faehig haben sich die in der Vorlesung behandelten Support Vector Maschinen
    # erwiesen.
    # Anhand der beiden bisher benutzten aber auch unter Verwendung eigener
    # Datensaetze soll nun die Mustererkennung mittels Support Vector Machines
    # untersucht werden.
    # Fuer Python stellt die Bibliothek 'sklearn' die Implementierung einer SVM
    # bereit.
    # http://scikit-learn.org/stable/modules/svm.html#svm
    # Zur Visualisierung der SVM kann die Funktion visualization.plot_svm
    # genutzt werden.

    # Der zweidimensionale Datensatz data2d enthaelt 3 Klassen.
    # Trainieren Sie zunaechst eine lineare Support Vector Maschine, die zur
    # Trennung von Klasse 0 und 2 verwendet werden soll. Klasse 1 wird hier
    # nicht betrachtet, da sie sich nicht linear von den Klassen 0 und 2 trennen
    # laesst (Siehe unten).
    # Wie hoch ist der Klassifikationsfehler im Vergleich zum Normalverteilungs-
    # klassifikator? Visualisieren Sie die resultierende Trennfunktion. Im Modul
    # visualization steht dafuer die Methode plot_svm bereit.
    # Diskutieren Sie den Einfluss der Slack-Variablen (in Form des C-Parameters)
    # auf die Trennfunktion und damit den entstehenden Klassifikationsfehler.
    #
    # Fuer die Evaluierung koennen Sie im Folgenden wieder den ClassificationEvaluator
    # aus dem Modul common.classifiers verwenden.

    raise NotImplementedError('Implement me')

    # Trainieren Sie nun eine SVM fuer die Klassen 1 und 2.
    # Evaluieren Sie, welcher Kernel geeignet ist, um das Problem zu loesen.
    raise NotImplementedError('Implement me')

    # Trainieren Sie nun eine Multi-Class SVM zur Loesung des 3-Klassenproblems
    # unter Verwendung eines geeigneten Kernels.
    # Wie koennen die optimalen kernelspezifischen Parameter sinnvoll ermittelt
    # werden?
    # Hinweis: Starten Sie zunaechst mit den Grundeinstellungen der Bibliothek.

    raise NotImplementedError('Implement me')

    # Vergleichen Sie Ihre Ergebnisse mit den bisher erzielten
    # Klassifikationsfehlerraten.


if __name__ == '__main__':
    aufg07()
