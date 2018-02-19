import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from common.data_provider import DataProvider
from common.pca import PCA


def aufg05():
    #
    # Realisieren Sie eine Hauptachsen-Transformation (PCA) in der Klasse PCA.
    # Zunaechst schauen wir uns das Prinzip an einem Spielbeispiel an.
    #
    # Mit der Methode pca_example wird die PCA anhand eines
    # 3D-Beispieldatensatzes visualisiert.
    # Nuetzliche Funktionen: np.linalg.eig
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    # Achten Sie darauf, dass die NumPy Methode np.linalg.eig komplexwertige Ergebnisse
    # liefern kann. Dies kann an numerischen Ungenauigkeiten liegen. Verwenden Sie in
    # diesem Fall nur den Realteil.
    pca_example()

    #
    # Nachdem bisher mit artifiziellen Datensaetzen gearbeitet wurde, wenden wir
    # uns jetzt realen Daten zu. Dazu verwenden wir den MNIST-Datensatz der
    # Grauwert-Bilder handgeschriebener Ziffern enthaelt. Der MNIST-Datensatz
    # besteht im Original aus 60000 Trainingsbildern und 10000 Testbildern. Um
    # den Aufwand geringer zu halten, werden im Rahmen dieser Uebung
    # lediglich 1000 zufaellig ausgewaehlte Trainingsbilder pro Klasse verwendet.
    #
    # Somit ergibt sich ein Trainingsdatensatz von 10000 sowie ein
    # Testdatensatz von 10000 Bildern.
    #
    # Die 28 x 28 Pixel grossen Bilder koennen als 784-dimensionale Merkmalsvektoren
    # aufgefasst werden.
    #
    # Laden Sie die Trainingsdaten des MNIST-Datensatz.
    # Das Laden des Datensatzes kann einige Sekunden in Anspruch nehmen.
    # Mit show_data(data, width) koennen Sie Bilder anzeigen lassen. Die Anzahl der
    # Bilder muss ein Vielfaches des Parameters width sein.
    train_data_provider = DataProvider(DataProvider.MNIST_TRAIN)
    raise NotImplementedError('Implement me')

    # Transformieren Sie die 784-dimensionalen Daten des MNIST-Datensatzes in
    # einen geeignet gewaehlten niedriger-dimensionalen Merkmalsraum. Begruenden
    # Sie die gewaehlte Groesse.
    # Hinweis: Die Unterraumdimension laesst sich mit der moeglichen
    # Rekonstruktionsqualitaet verknuepfen.
    # Es ist empfehlenswert, die dimensionsreduzierten Daten fuer die spaetere
    # Nutzung zu speichern.
    # Nuetzliche Funktion: DataProvider.write_data oder pickle
    # Achten Sie darauf, dass DataProvider.write_data ein dictionary als
    # Eingabe erwartet.
    # Zu pickle finden Sie weitere Informationen hier:
    # https://docs.python.org/2/library/pickle.html
    # https://wiki.python.org/moin/UsingPickle

    # Optional: Visualisieren Sie die MNIST-Daten in einem 2D Unterraum. Faerben Sie
    # die Datenpunkte nach Klassenzugehoerigkeit.

    raise NotImplementedError('Implement me')


def show_data(data, width=1):
    '''
    Stellt die Bilder in data zeilenweise da. Nebeneinander werden width-viele
    Bilder angezeigt. Die Gesamtanzahl der Bilder muss so gewaehlt sein, dass in jeder
    Zeile width-viele Bilder dargestellt werden koennen.
    Params:
        data: Darzustellende Bilder als 2D-ndarray. Eine Zeile entspricht einem Bild.
        width: Anzahl der Bilder einer Zeile in der Visualisierung. (default = 1)
    '''
    if len(data.shape) == 1:
        data = data.reshape(1, data.shape[0])
        image_count = 1
    else:
        image_count = data.shape[0]

    image = []
    for i in np.arange(width):
        index = np.arange(i, image_count, width)
        column = data[index, :]
        image.append(column.reshape((28 * column.shape[0], 28)))
    image = np.hstack(tuple(image))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap=cm.get_cmap('Greys_r'))
    ax.set_xticks([])
    ax.set_yticks([])


def pca_example():
    #
    # Ein 3D Beispieldatensatz wird aus einer Normalverteilung generiert.
    # Diese ist durch einen Mittelwert und eine Kovarianzmatrix definiert
    mean = np.array([10, 10, 10])
    cov = np.array([[3, .2, .9],
                    [.2, 5, .4],
                    [.9, .4, 9]])
    n_samples = 1000
    limits_samples = ((0, 20), (0, 20), (0, 20))
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    # Plotten der Beispieldaten
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    PCA.plot_sample_data(samples, ax=ax)
    PCA.set_axis_limits(ax, limits=limits_samples)

    #
    # In der Klasse PCA wird ein Unterraum mittels Hauptkomponentenanalyse
    # statistisch geschaetzt. Der Vektorraum wird beispielhaft visualisiert.
    # Implementieren Sie zunaechst den Konstruktor der Klasse PCA.
    pca_tutorial = PCA(samples)
    pca_tutorial.plot_subspace(limits=limits_samples, color='r',
                               linewidth=0.05, alpha=0.3)

    #
    # Transformieren Sie nun die 3D Beispieldaten in den 2D Unterraum.
    # Implementieren Sie dazu die Methode transform_samples. Die Daten werden
    # dann in einem 2D Plot dargestellt.
    #
    # Optional: Erweitern Sie die Funktion plot_subspace, so dass die Eigenvektoren und
    # Eigenwerte des 2D Unterraums verwendet werden. Verwenden Sie den optionalen Parameter
    # target_dim.


    samples_2d = pca_tutorial.transform_samples(samples, target_dim=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    PCA.plot_sample_data(samples_2d, ax=ax)
    PCA.set_axis_limits(ax, limits=((-10, 10), (-10, 10)))
    plt.show()

    #
    # Unten wird die Kovarianzmatrix der transformierten Daten berechnet.
    # Welche Eigenschaften hat diese Matrix? (Dimension, etc.)
    # In welcher Groessenordnung liegen die einzelnen Eintraege? Erklaeren Sie das
    # anhand des vorherigen 2D Plots.
    # Vergleichen Sie das Ergebnis mit der Kovarianzmatrix, die oben zur Generierung
    # der Daten verwendet wurde.
    # Erklaeren Sie den Mittelwert der transformierten Daten.

    samples_2d_mean = np.sum(samples_2d, axis=0)
    samples_2d_meanfree = samples_2d - samples_2d_mean
    samples_2d_cov = np.dot(samples_2d_meanfree.T, samples_2d_meanfree) / samples_2d.shape[0]
    print 'samples_2d mean'
    print samples_2d_mean
    print 'samples_2d covariance matrix'
    print samples_2d_cov


if __name__ == '__main__':
    aufg05()
