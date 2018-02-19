from common.data_provider import DataProvider
from common.classifiers import KNNClassifier 
from common.evaluation import ClassificationEvaluator
from common.evaluation import CrossValidation
import matplotlib.pyplot as plt



def aufg01():
    # Zur Einfuehrung werden auf einem Beispieldatensatz Klassifikatoren
    # implementiert. Der Datensatz data2d enthaelt zweidimensionalen
    # Trainingsmerkmale fuer drei Musterklassen. Fuer diese Daten soll eine
    # Klassifikation ueber Naechster-Nachbar realisiert  werden.
    # Achtung: Gestalten Sie Ihre Implementierung so, dass Sie die Klassifikatoren
    # fuer zukuenftige Aufgaben wiederverwenden koennen.

    # Im Folgenden wird der Beispieldatensatz ueber die Klasse DataProvided
    # geladen und anschliessend visualisiert. Machen Sie sich mit sowohl dem
    # Laden als auch der Visualisierung vertraut, da Sie in den kommenden
    # Aufgaben diese Aspekte wiederverwenden werden.
    # http://matplotlib.org/users/pyplot_tutorial.html
    #
    # Nuetzliche Funktionen: plt.scatter
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
    #
    # Tipp: zu einer besseren Visualisierung sollten alle scatter plots in Matplotlib
    # immer mit dem Argument "edgecolor=(0, 0, 0)" aufgerufen werden.

    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = train_data_provider.get_class_arr(0)
    ax.scatter(data[:, 0], data[:, 1], c='#FF0000', edgecolor=(0, 0, 0))
    data = train_data_provider.get_class_arr(1)
    ax.scatter(data[:, 0], data[:, 1], c='#00FF00', edgecolor=(0, 0, 0))
    data = train_data_provider.get_class_arr(2)
    ax.scatter(data[:, 0], data[:, 1], c='#0000FF', edgecolor=(0, 0, 0))

    plt.show()

    # Implementieren Sie einen Naechster-Nachbar-Klassifikator.
    # Vervollstaendigen Sie dazu die Klasse KNNClassifier im Modul common.classifiers.
    # Testen Sie verschiedene Abstandsmasse. Welche halten Sie insbesondere fuer sinnvoll?
    train_data, train_labels = train_data_provider.get_dataset_and_labels()
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()
    knn_classifier = KNNClassifier(k_neighbors=29, metric='euclidean')
    knn_classifier.estimate(train_data, train_labels)
    estimated_labels = knn_classifier.classify(test_data)

    

    # Nutzen Sie zur Evaluation der Ergebnisse die Klasse ClassificationEvaluator
    # im Modul common.classifiers.

    evaluator = ClassificationEvaluator(estimated_labels, test_labels_gt)

    print("Error rate {}".format(evaluator.error_rate()))
    print("Category error rate {}".format(evaluator.category_error_rates()))
    
    # Ein NN-Klassifikator alleine ist meist nicht ausreichend. Erweitern Sie
    # den Klassifikator zum k-NN Klassifikator.
    # Fuer den Mehrheitsentscheid ist das defaultdict nuetzlich (siehe intro).
    # https://docs.python.org/2/library/collections.html#collections.defaultdict

    # Trainingsparameter sollten immer per Kreuzvalidierung auf den Trainingsdaten
    # optimiert werden. Mit den besten Parametern wird dann ein Klassifikator
    # erstellt und auf den Testdaten evaluiert.
    # Nutzen Sie die Klasse CrossValidation im Modul classification um den
    # Parameter k zu optimieren.
    # In den folgenden Aufgaben ist es Ihnen freigestellt, ob Sie Kreuzvalidierung
    # nutzen oder direkt auf den Testdaten optimieren.

    cross_vali = CrossValidation(train_data, train_labels, 5)

    over_all_result, class_result = cross_vali.validate(
            KNNClassifier(k_neighbors=10, metric='euclidean'))
    print(over_all_result)


if __name__ == '__main__':
    aufg01()
