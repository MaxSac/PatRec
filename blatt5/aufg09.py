import matplotlib.pyplot as plt
import numpy as np

from common import visualization
from common.classifiers import Perceptron
from common.nn_loss import EuclideanLoss

#
# In dieser und der naechsten Aufgabe geht es um die Klassifikation von Daten mit
# neuronalen Netzen. Zunaechst soll in dieser Aufgabe ein Perceptron
# implementiert werden, waehrend in der folgende Aufgabe 10 ein Multilayer Perceptron
# verwendet werden wird. Da das Training von beiden neuronalen Netzen einem
# gleichem Schema folgt, sollen Sie als Teil dieser Aufgabe eine teilweise
# implementierte, abstrakte Basisklasse vervollstaendigen, die als Grundlage
# fuer sowohl das Perceptron als auch des MLP dienen soll.
# Im Sinne guter Softwareentwicklung, werden Perceptron und MLP modular implementiert.
# Die wesentlichen Elemente beider neuronalen Netze, die auch zwischen beiden
# geteilt werden, sind die Schichten sowie die verwendeten Loss-Funktionen.
# Diese sollen im Folgenden zunaechst fuer das Perceptron implementiert werden
# und dann fuer das MLP wiederverwendet werden.
#
# Ein Perceptron ist ein einschichtiges neuronales Netz ohne versteckte Schichten.
# Es kann daher als Kombination aus einer vollvernetzten Schicht (Fully Connected
# Layer) und der gewuenschten Aktivierungsfunktion implementiert werden.
# Dabei wird typischerweise die Aktivierungsfunktion als eigene Schicht implementiert,
# um sie einfach einsetzen zu koennen. Beachten Sie jedoch, dass dies lediglich
# eine programmiertechnische Vereinfachung ist. Konzeptionell sind Aktivierungs-
# funktionen KEINE eigenen Schichten, sondern gehoeren immer zu einem Neuron.
#
# Implementieren Sie die Fully Connected Schicht im Modul nn_layers.py.
# Implementieren Sie ausserdem die Euclidean Loss Funktion im Modul nn_loss.py.
# Fahren Sie erst dann mit der Implementierung der Perceptron-Klasse
# im Modul classifiers fort.
#
# Tipp: Sie koennen die Korrektheit der Funktionalitaet der zu implementierenden
# Klasse mit Hilfe der Unittests in tests/perceptron_tests.npy ueberpruefen.
#
#
#
# Beantworten Sie zusaetzlich die folgenden Fragen:
#
# Welche Klassifikationsprobleme lassen sich mit einem Perceptron loesen?
# Welchen Zweck erfuellt das bias-Gewicht?
# Wie wirken sich unterschiedliche Lernraten aus?
# Warum springt die Hyperebene beim Training teilweise zurueck zu einem schlechteren Ergebnis?
# Wie verhalten sich die Ergebnisse im Vergleich zu einer linearen 2-Klassen SVM?
#
#
# Nachdem das Perceptron implementiert wurde, muss nun noch der Trainingsprozess
# implementiert werden. Dieser ist, wie bereits erwaehnt, fuer Perceptron und MLP
# gleich. Daher soll er in einer abstrakten Basisklasse implementiert werden,
# sodass er spaeter einfach fuer beide neuronalen Netze verwendet werden kann.
# Implementieren Sie die fehlenden Abschnitte in der Klasse NeuralNetworkBase
# im Modul classifiers.py


def aufg09():
    #
    # Nachdem das Perceptron komplett implementiert wurde, wird es im folgenden
    # trainiert und ausgewertet. Zunaechst wird dafuer ein Datensatz erzeugt.
    train_data = []
    train_data.append(np.random.multivariate_normal((9, 9), ((1, 0), (0, 1)), size=1000))
    train_data.append(np.random.multivariate_normal((3, 3), ((1, 0), (0, 1)), size=1000))
    train_data = np.vstack(train_data)
    train_labels = np.hstack((np.zeros(1000, dtype=np.uint8), np.ones(1000, dtype=np.uint8)))
    test_data = train_data
    test_labels = train_labels

    #
    # Zum Training muessen die die Labels als One-Hot Vektoren dargestellt werden.
    #
    # Was sind One-Hot Encodings und warum werden sie fuer neuronale Netze benoetigt?
    class_labels = np.unique(train_labels)
    one_hot_train_labels = train_labels.reshape((-1, 1)) == class_labels
    one_hot_train_labels = one_hot_train_labels.astype(np.float32)

    #
    # Das Perceptron wird mit einer linearen Aktivierungsfunktion
    # und dem EuclideanLoss als Loss-Funktion initialisiert.
    perceptron = Perceptron(n_input=2,
                            n_output=2,
                            activation_function=None,
                            loss=EuclideanLoss(),
                            batch_size=100,
                            epochs=1,
                            learning_rate=1e-4,
                            momentum=0.0)

    #
    # Das Perceptron wird verwendet, um einmal die Daten zu klassifizieren,
    # bevor das Training beginnt.
    pred_labels = perceptron.classify(test_data)
    before_acc = np.sum(pred_labels == test_labels) / float(test_labels.shape[0])
    print 'Accuracy before training: %f' % before_acc
    fig = plt.figure()
    plt.title('Hyperebene vor Training')
    ax = fig.add_subplot(111)
    visualization.plot_hyperplane(ax, test_data, test_labels, predict_func=perceptron.classify)
    plt.show()

    # Training
    plt.ion()
    fig = plt.figure()
    plt.title('Training...')
    ax = fig.add_subplot(111)
    for idx in range(30):
        perceptron.estimate(train_samples=train_data,
                            train_labels=one_hot_train_labels)
        visualization.plot_hyperplane(ax, test_data, test_labels, predict_func=perceptron.classify)
        plt.pause(0.01)
        plt.draw()

        # Klassifikation der Daten nach einer Epoche
        pred_labels = perceptron.classify(test_data)
        acc = np.sum(pred_labels == test_labels) / float(test_labels.shape[0])
        print 'Accuracy after %*d training epochs: %f' % (4, (idx + 1) * 100, acc)


if __name__ == '__main__':
    aufg09()
