import matplotlib.pyplot as plt
import numpy as np

from common import visualization
from common.classifiers import MultilayerPerceptron
from common.nn_layers import SigmoidLayer
from common.nn_loss import BinaryCrossEntropyLoss

#
# In dieser Aufgabe sollen Sie, aufbauend auf Ihren bisherigen Implementierungen
# ein Multilayer Perceptron (MLP) erstellen und dieses trainieren.
# Die notwendigen Methoden aus der Klasse NeuralNetworkBase wurden in der vor-
# hergehenden Aufgabe bereits implementiert und koennen daher wieder verwendet
# werden.
#
# Fuer das MLP benoetigen Sie zusaetzlich zu der bisher implementierten Fully
# Connected Schicht die Schicht SigmoidLayer. Ausserdem benoetigen Sie die
# Loss-Funktion BinaryCrossEntropyLoss. Implementieren Sie zunaechst die
# Schicht und den Loss im Modul nn_layers.py bzw. nn_loss.py.
#
# Implementieren Sie anschliessend die fehlenden Abschnitte Klasse MultilayerPerceptron
# im Modul classifiers.
#
#
# Tipp: Sie koennen die Korrektheit der Funktionalitaet der zu implementierenden
# Klasse mit Hilfe der Unittests in tests/mlp_tests.npy ueberpruefen.
#
#
# Beantworten Sie zusaetzlich die folgenden Fragen:
#
# Warum sollten MLPs nicht zur Klassifikation von Bildern verwendet werden?
# Warum ist es sinnvoll als Aktivierungsfunktion der versteckten Schichten die
# Rectified Linear Unit zu verwenden?


def aufg10():
    #
    # Nachdem Sie das MLP und die noetigen Schichten implementiert haben,
    # koennen Sie es nun trainieren. Zunaechst werden wieder Trainings- und
    # Testdaten erzeugt...
    train_data = []
    train_data.append(np.random.multivariate_normal((4, 4), ((1, 0), (0, 1)), size=500))
    train_data.append(np.random.multivariate_normal((-4, -4), ((1, 0), (0, 1)), size=500))
    train_data.append(np.random.multivariate_normal((-1, 1), ((1, 0), (0, 1)), size=1000))
    train_data = np.vstack(train_data)
    train_labels = np.hstack((np.zeros(1000, dtype=np.uint8), np.ones(1000, dtype=np.uint8)))
    test_data = train_data
    test_labels = train_labels
    # ... und diese in ein One-Hot Encoding transformiert
    class_labels = np.unique(train_labels)
    one_hot_train_labels = train_labels.reshape((-1, 1)) == class_labels
    one_hot_train_labels = one_hot_train_labels.astype(np.float32)

    #
    # Anschliessend wird ein MLP-Objekt erzeugt...
    mlp = MultilayerPerceptron(n_input=2, n_hidden=[100, 100], n_output=2,
                               activation_function_hidden=SigmoidLayer,
                               activation_function_output=SigmoidLayer,
                               loss=BinaryCrossEntropyLoss(),
                               batch_size=100,
                               epochs=1,
                               learning_rate=1e-2,
                               momentum=0.0)
    # .. welches vor dem Training einmal ausgewertet wird.
    pred_labels = mlp.classify(test_data)
    before_acc = np.sum(pred_labels == test_labels) / float(test_labels.shape[0])
    print 'Accuracy before training: %f' % before_acc
    fig = plt.figure()
    plt.title('Class Boundary vor Training')
    ax = fig.add_subplot(111)
    visualization.plot_hyperplane(ax, test_data, test_labels, predict_func=mlp.classify)
    plt.show()

    # Als letztes kommt das eigentliche Training.
    plt.ion()
    fig = plt.figure()
    plt.title('Training...')
    ax = fig.add_subplot(111)
    for idx in range(50):
        mlp.estimate(train_samples=train_data,
                     train_labels=one_hot_train_labels)
        visualization.plot_hyperplane(ax=ax,
                                      data=test_data,
                                      labels=test_labels,
                                      predict_func=mlp.classify)
        plt.pause(0.01)
        plt.draw()

        # classify again after training
        pred_labels = mlp.classify(test_data)
        acc = np.sum(pred_labels == test_labels) / float(test_labels.shape[0])
        print 'Accuracy after %*d training epochs: %f' % (4, (idx + 1) * 100, acc)

    #
    # Implementieren Sie die Schicht ReLU Layer in der Datei layers.py und
    # verwenden Sie sie als Aktivierungsfunktion der versteckten Schichten.    #
    # Veraendern Sie die Learnrate, Momentum und die Groesse und Anzahl der
    # versteckten Schichten. Was faellt Ihnen auf?


if __name__ == '__main__':
    aufg10()
