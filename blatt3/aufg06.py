from common.data_provider import DataProvider


def aufg06():
    #
    # Nun sollen die Daten des MNIST-Datensatzes mit den vorher erstellten
    # Klassifikatoren klassifiziert werden. Trainieren Sie dazu jeweils mit den
    # Trainigsdaten einen Klassifikator und berechnen Sie den sich ergebenden
    # Klassifikationsfehler auf dem Testdatensatz. Variieren Sie die
    # Parametrisierung der Klassifikatoren,
    # um einen moeglichst geringen Klassifikationsfehler zu erreichen.
    #
    # Trainieren Sie einen Mischverteilungsklassifikator fuer verschiedene mit
    # PCA dimensionsreduzierte Versionen des MNIST-Datensatzes und vergleichen
    # Sie die erzielten Ergebnisse.
    #
    # Trainieren Sie einen k-NN-Klassifikator fuer verschiedene mit PCA
    # dimensionsreduzierte Versionen des MNIST-Datensatzes und vergleichen Sie
    # die erzielten Ergebnisse.

    train_data_provider = DataProvider(DataProvider.MNIST_TRAIN_PCA)
    test_data_provider = DataProvider(DataProvider.MNIST_TEST_PCA)
    train_data, train_labels = train_data_provider.get_dataset_and_labels()
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()

    raise NotImplementedError('Implement me')


if __name__ == '__main__':
    aufg06()
