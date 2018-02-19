from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from common import visualization
from common.classifiers import MDClassifierClassIndep, MDClassifierClassDep
from common.data_provider import DataProvider
from common.evaluation import ClassificationEvaluator
from common.vector_quantization import Lloyd


def aufg04():
    #
    # Nutzen Sie den in Aufgabe 3 Vektorquantisierer fuer die Schaetzung eines
    # Gaussian Mixture Models.
    #
    # Implementieren Sie dazu die Klasse MDClassifierClassIndep im Modul common.classifiers.
    #
    # Welche zusaetzlichen Parameter werden neben den Mischverteilungsparametern
    # noch fuer die Klassifizierung benoetigt?
    #
    # Werten Sie Ihre Implementierung mit dem unten stehenden Code aus.

    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    train_data, train_labels = train_data_provider.get_dataset_and_labels()
    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()

    quant = Lloyd()
    classifier = MDClassifierClassIndep(quant, 10)
    classifier.estimate(train_data, train_labels)

    estimated_labels = classifier.classify(test_data)
    evaluator = ClassificationEvaluator(estimated_labels, test_labels_gt)
    print 'Klassenunabhaengig:'
    print 'Fehlerrate: %.1f; Anzahl falsch-klassifizierte Muster: %d; Anzahl Muster: %d' % evaluator.error_rate()
    print 'Klassenspezifische Fehlerraten'
    for res in evaluator.category_error_rates():
        print 'Klasse %s:\tFehlerrate: %.1f;\tAnzahl falsch-klassifizierte Muster: %d;\tAnzahl Muster: %d' % res

    cmap = cm.get_cmap('hsv')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean = classifier.mean
    cov = classifier.cov
    c = cmap(np.linspace(0, 1, len(cov)))
    visualization.plot_norm_dist_ellipse(ax, mean, cov, c)
    ax.scatter(train_data[:, 0], train_data[:, 1], c='#FFFFFF', edgecolor=(0, 0, 0))

    # Realisieren Sie den Mischverteilungsklassifikator mit klassenabhaengigen
    # Komponentendichten. Implementieren Sie dazu die Klasse MDClassifierClassDep
    # im Modul common.classifiers.

    classifier = MDClassifierClassDep(quant, (1, 3, 1))
    classifier.estimate(train_data, train_labels)

    estimated_labels = classifier.classify(test_data)
    evaluator = ClassificationEvaluator(estimated_labels, test_labels_gt)
    print '\n##################################################\n'
    print 'Klassenabhaengig'
    print 'Fehlerrate: %.1f; Anzahl falsch-klassifizierte Muster: %d; Anzahl Muster: %d' % evaluator.error_rate()
    print 'Klassenspezifische Fehlerraten'
    for res in evaluator.category_error_rates():
        print 'Klasse %s:\tFehlerrate: %.1f;\tAnzahl falsch-klassifizierte Muster: %d;\tAnzahl Muster: %d' % res

    classes = classifier.classifier
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for index, c in enumerate(classes):
        cov = c.cov
        col = [cmap(index / float(len(classes)))] * len(cov)
        visualization.plot_norm_dist_ellipse(ax, c.mean, cov, col)
        data = train_data_provider.get_class_arr(index)
        ax.scatter(data[:, 0], data[:, 1], c=col, edgecolor=(0, 0, 0))
    plt.show()


if __name__ == '__main__':
    aufg04()
