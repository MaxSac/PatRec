import numpy as np
import scipy.spatial.distance
from collections import defaultdict
from scipy.spatial.distance import cdist
from common import log_math
from common.nn_layers import FullyConnectedLayer
#--------------------------------------------------


###############################################################################
#                   NAECHSTER NACHBAR KLASSIFIKATOR                           #
#                              Aufg. 1                                        #
###############################################################################
class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
        '''
        Initialisiert den Klassifikator mit Meta-Parametern

        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string),
                siehe auch scipy Funktion cdist
        '''
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.train_samples = None
        self.train_labels = None

        # raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        '''
        Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.

        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden.
        Diskutieren Sie, was den K-NN stattdessen definiert.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        '''

        self.train_samples = train_samples
        self.train_labels = train_labels
        
        # raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        '''
        Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        '''

        # Implementieren Sie die Klassifikation der Daten durch den KNN.
        #
        # Nuetzliche Funktionen: scipy.spatial.distance.cdist, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

        distance = cdist(test_samples, self.train_samples, metric=self.metric)
        sort = np.argsort(distance)
        knn_labels = self.train_labels[sort][:,:self.k_neighbors]
        
        knn_mean_labels = np.array([])
        for knn_label in knn_labels:
            d= defaultdict(int)
            for y in knn_label:
                d[y] += 1
            knn_mean_labels = np.append(knn_mean_labels, max(d, key=d.get))

        return knn_mean_labels




###############################################################################
#                     STATISTISCHE KLASSIFIKATOREN                            #
#                          Aufg. 2 - 4                                        #
###############################################################################
class GaussianClassifier(object):

    def __init__(self):
        '''
        Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.
        '''
        self.means = []
        self.covs = []
        self.a_priori = []
        
        #raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        '''
        Erstellt den Normalverteilungsklassikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        self.labels = np.unique(train_labels)
        self.means=[]
        self.covs=[]
        self.a_priori = []
        for label in self.labels:
            
            class_data = train_samples[train_labels==label]
            mean = np.mean(class_data, axis=0)
            cov = np.cov(class_data, rowvar=0)
            a_priori = np.float32(len(class_data))/np.float32(len(train_samples))

            self.means.append(mean)
            self.covs.append(cov)
            self.a_priori.append(a_priori)


        #raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        '''Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        
        # Werten Sie die Dichten aus und klassifizieren Sie die
        # Testdaten.
        #
        # Hinweise:
        #
        # Durch welche geeignete monotone Transformation lassen sich numerische
        # Probleme bei der Auswertung von extrem kleinen Dichtewerten vermeiden?
        # Beruecksichtigen Sie das in Ihrer Implementierung.
        #
        # Erstellen Sie fuer die Auswertung der transformierten Normalverteilung
        # eine eigene Funktion. Diese wird in den folgenden Aufgaben noch von
        # Nutzen sein.
        
        def multivariate_normal(x, mean, cov, a_priori=1):
            d = len(mean)
            return np.log(a_priori) - 0.5*np.log((2*np.pi)**d*np.linalg.det(cov))   \
                    - 0.5*np.diag(np.dot((x - mean), np.dot(np.linalg.inv(cov),     \
                    np.transpose(x - mean))))

        pdf = []
        for mean, cov, a_priori in zip(self.means, self.covs, self.a_priori):
            pdf.append(multivariate_normal(test_samples, mean, cov, a_priori))

        pdf_label = np.argmax(pdf, axis=0)
        
        return self.labels[pdf_label]

        # raise NotImplementedError('Implement me')


class MDClassifierClassIndep(object):

    def __init__(self, quantizer, num_densities):
        '''Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die verwendet
                werden sollen.
        '''
        self.quantizer = quantizer
        self.num_densities = num_densities
        raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        '''Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (*d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        #
        # Diese Methode soll das Training eines Mischverteilungsklassifikators
        # mit klassenunabhaengigen Komponentendichten implementieren (siehe Skript S. 67 f.).
        #
        # Die folgenden Funtionen koennen bei der Implementierung von Nutzen
        # sein:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.slogdet.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html

        #
        # Schaetzen Sie das GMM der Trainingsdaten.
        #
        # Wieviele Datenpunkte werden zur Schaetzung jeder Normalverteilung mindestens
        # benoetigt und welche Eigenschaften muessen diese haben?
        # Beruecksichtigen Sie das in Ihrer Implementierung.

        raise NotImplementedError('Implement me')

        #
        # Bestimmen Sie fuer jede Klasse die spezifischen Mischungsgewichte.
        # Beachten Sie, dass die Dichteauswertung wieder ueber eine geeignete
        # monotome Transformationsfunktion geschehen soll. Verwenden Sie hierfuer
        # die Funktion, die Sie bereits fuer den GaussianClassifier implementiert
        #
        # Achten Sie darauf, dass sich die Mischungsgewichte zu 1 addieren.

        raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        '''Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        raise NotImplementedError('Implement me')


class MDClassifierClassDep(object):

    def __init__(self, quantizer, num_densities):
        '''Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die je Klasse
                verwendet werden sollen.
        '''
        raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        '''Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        #
        # Schaetzen Sie die jeweils ein GMM fuer jede Klasse.
        #
        # Tipp: Verwenden Sie die bereits implementierte Klasse MDClassifierClassIndep

        raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        '''Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        '''
        raise NotImplementedError('Implement me')

###############################################################################
#                           NEURONALE NETZE                                   #
#                            Aufg. 9 - 10                                     #
###############################################################################


class NeuralNetworkBase(object):
    '''
    Abstrakte Basisklasse fuer alle neuronalen Netze

    Diese Klasse dient als Grundlage fuer alle neuronalen Netze, die Sie in der
    Uebung implementieren werden.
    Machen Sie sich mit dieser Klasse vertraut.
    '''

    def __init__(self, loss, batch_size, epochs, learning_rate, momentum):
        '''
        Konstruktor

        Zum initialisieren des neuronalen Netzes wird ein Loss-Object uebergeben,
        welches als Klassenvariable gespeichert wird. Auf dieses Loss-Object
        wird spaeter waehrend des Trainings zugegriffen.
        Als Loss-Objekte sollen Instanzen der Loss-Klassen verwendet werden,
        die in der Datei loss.py implementiert sind.
        '''
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, data):
        '''
        Abstrakte Methode fuer den Forward Pass

        Diese Methode muss in einer entsprechenden Unterklasse ueberschrieben
        werden und um die gewuenschte Funktionalitaet erweitert werden.
        '''
        raise NotImplementedError

    def backward(self, top_gradient):
        '''
        Abstrakte Methode fuer den Backward Pass

        Diese Methode muss in einer entsprechenden Unterklasse ueberschrieben
        werden und um die gewuenschte Funktionalitaet erweitert werden.
        '''
        raise NotImplementedError

    def apply_update(self, learning_rate, momentum):
        '''
        Abstrakte Methode fuer den Gradientenabstieg

        Diese Methode muss in einer entsprechenden Unterklasse ueberschrieben
        werden und um die gewuenschte Funktionalitaet erweitert werden.
        '''
        raise NotImplementedError

    def estimate(self, train_samples, train_labels):
        '''
        Trainiert das neuronale Netz

        Diese Methode wird aufgerufen, wenn das neuronale Netz trainiert werden
        soll. Auf Grund der generellen Struktur von neuronalen Netzen, ist
        diese Methode fuer alle Typen von feed-forward Netzen gleich.

        :param train_samples: die Daten, mit denen das Netz trainiert werden soll
        :param labels: die Annotation der Daten
        :param batch_size: die Batch Size, die fuer den SGD verwendet werden soll
        :param epochs: die Anzahl an Epochen, die trainiert werden soll
        :param learning_rate:
        :param momentum:
        '''
        #
        # Als erstes werden eine Reihe von Checks durchgefuehrt, um die Daten-
        # konsistenz zu gewaehrleisten
        if train_samples.shape[0] % self.batch_size != 0:
            raise ValueError('Die Anzahl der Trainings Samples Modulo Batch Size muss 0 ergeben.')
        if train_samples.shape[0] != train_labels.shape[0]:
            raise ValueError('Anzahl der Trainings Samples und der Labels muss uerbereinstimmen.')

        #
        # Bevor ein neuronales Netz trainiert wird, sollten die Daten auf eine
        # bestimme Weise vorverarbeitet werden.
        # Wie sollte die Vorverarbeitung aussehen? Warum ist sie sinnvoll?
        #
        # Implementieren Sie die Vorverarbeitung.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.shuffle.html

        raise NotImplementedError('Implement me')

        #
        # Im Folgenden wird die Trainingsmenge in Mini-Batches zerteilt,
        # die spaeter fuer den SGD verwendet werden.
        n_batches = train_samples.shape[0] / self.batch_size
        batches_train_samples = np.split(train_samples, n_batches, axis=0)
        batches_labels = np.split(train_labels, n_batches, axis=0)

        #
        # Die folgende Schleife enthaelt das Training des neuronalen Netzes
        # mit Hilfe von Gradientenabstieg
        for epoch_idx in range(self.epochs):
            #
            # Die jeweiligen Loss-Werte der einzelnen Mini-Batches werden
            # aufsummiert, um so den totalen Loss der derzeitigen Epoche
            # feststellen zu koennen. Dafuer wird am Anfang jeder Epoche eine
            # Zaehlvariable eingerichtet.
            total_loss = 0

            #
            # Eine Epoche beinhaltet dann, dass alle Mini-Batches einmal durchlaufen
            # werden und die Gewichte nach jedem Mini-Batch entsprechend angepasst
            # werden.
            for batch_x, batch_y in zip(batches_train_samples, batches_labels):
                #
                # Zum Training mit Gradientenabstieg wird ein Forward Pass
                # ausgefuehrt, um die Ausgabe des neuronalen Netzes zu bestimmen.
                # Fuehren Sie diesen Forward Pass aus und speichern Sie das Ergebnis
                # in der lokalen Variablen y_pred.
                # Benutzen Sie die abstrakte Methode forward, die spaeter in den
                # Unterklassen um ihre entsprechende Funktionalitaet ergaenzt wird.

                raise NotImplementedError('Implement me')

                #
                # Mit der Netzausgabe und der gewuenschten Ausgabe kann dann
                # der Loss bestimmt werden.
                total_loss += self.loss(y_pred=y_pred, y_label=batch_y)

                #
                # Bestimmen Sie den Gradienten des Loss ueber die gradient
                # Methode des Loss-Objects und speichern Sie ihn in der lokalen
                # Variablen gradient_loss.

                raise NotImplementedError('Implement me')

                #
                # Fuehren Sie eine backward pass des Gradienten durch das
                # neuronale Netz durch. Verwenden Sie hierfuer den Gradienten
                # des Loss (gradient_loss).

                raise NotImplementedError('Implement me')

                #
                # Nach dem backward pass werden die Gewichte der entsprechenden
                # Schichten veraendert, so dass ein Gradientenabstieg geschieht.
                self.apply_update(learning_rate=self.learning_rate,
                                  momentum=self.momentum)

            # Zum Schluss wird noch der totale Loss ausgegeben, der in dieser
            # Epoche beobachtet wurde.
            if (epoch_idx + 1) % 100 == 0:
                print 'Loss after %d epochs: %f' % (epoch_idx + 1, total_loss)

    def classify(self, test_samples):
        '''
        Klassifikation der Eingabedaten

        Behandelt die Ausgabe als Klassifikationsergebnis und gibt die vorher-
        gesagte Klasse zurueck. Diese Methode wird zur Anzeige benoetigt.

        :param data (ndarray): Eingabedaten, die klassifiziert werden sollen
                               Groesse: (n_samples, n_dimensions)
        '''
        #
        # Klassifizieren Sie die Daten in test_samples mit dem neuronalen Netz.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html

        raise NotImplementedError('Implement me')


class Perceptron(NeuralNetworkBase):

    def __init__(self, n_input, n_output, activation_function, loss,
                 batch_size, epochs, learning_rate, momentum):
        '''
        Konstruktor

        :param n_input: Groesse der Eingabedaten (Dimensionen)
        :param n_output: Ausgabegroesse (Dimensionen)
        :param activation_function: Die Aktivierungsfunktion die verwendet werden soll
        '''
        super(Perceptron, self).__init__(loss=loss, batch_size=batch_size, epochs=epochs,
                                         learning_rate=learning_rate, momentum=momentum)
        #
        # Das Perceptron besteht aus einem Fully Connected Layer
        # und einer Aktivierungsfunktion. Die Aktivierungsfunktion
        # wird als Objekt uebergeben. Legen Sie einen Fully Connected
        # Layer in der Klassenvariablen self.fc an.
        # Bei der Initialisierung des Perceptrons wird ein Objekt einer
        # Aktivierungsfunktion uebergeben, welche vom Benutzer gewuenscht
        # ist (siehe Aktivierungsfunktionen in der Datei nn_layers.py).
        # Speichern Sie das Aktivierungsfunktionsobjekt in der
        # Klassenvariablen self.activation_function

        raise NotImplementedError('Implement me')

    def forward(self, x_input):
        ''' Berechne die Ausgabe des Perceptrons (Forward Pass) '''
        #
        # Berechnen Sie den Forward Pass durch das Perceptron.
        # Beachten Sie, dass die Aktivierungsfunktion None
        # ist, wenn eine lineare Aktivierung gewuenscht ist.

        raise NotImplementedError('Implement me')

    def backward(self, top_gradient):
        ''' Berechne die Gradienten der Gewichte und der Eingaben (Backward Pass) '''
        #
        # Fuehren Sie einen Backward Pass durch das Perceptron durch.
        # Propagieren Sie dafuer den Fehler aus top_gradient zunaechst
        # rueckwaerts durch die verwendete Aktivierungsfunktion.
        # Anschliessend soll der entsprechende Gradient durch die Fully
        # Connected Schicht propagiert werden.
        # Beachten Sie, dass die Aktivierungsfunktion None
        # ist, wenn eine lineare Aktivierung gewuenscht ist.

        raise NotImplementedError('Implement me')

    def apply_update(self, learning_rate, momentum):
        '''
        Ein Schritt Gradientenabstieg mit Momentum

        Die im backward Aufruf berechneten Gradienten werden verwendet, um die
        Gewichte zu aktualisieren. Dies entspricht einem Schritt im
        Gradientenabstieg.
        '''
        self.fc.apply_update(learning_rate, momentum)


class MultilayerPerceptron(NeuralNetworkBase):

    def __init__(self, n_input, n_hidden, n_output,
                 activation_function_hidden,
                 activation_function_output,
                 loss,
                 batch_size, epochs, learning_rate, momentum):
        '''
        Konstruktor

        :param n_input (int): Groesse der Eingabedaten (Dimensionen)
        :param n_hidden (list of int): Liste der Groessen der versteckten Schichten
        :param n_output (int): Ausgabegroesse (Dimensionen)
        :param activation_function_hidden: Die Aktivierungsfunktion der versteckten Schichten
        :param activation_function_output: Die Aktivierungsfunktion der Ausgabeschicht
        '''
        super(MultilayerPerceptron, self).__init__(loss=loss, batch_size=batch_size, epochs=epochs,
                                                   learning_rate=learning_rate, momentum=momentum)
        # init the layers
        self.layers = []
        last_n = n_input
        # Fuegen Sie die Anzahl der gewuenschten versteckten Schichten ein.
        #
        # Aus welchen der zu implementierenden Schichten besteht eine versteckte
        # Schicht?
        #
        # Das argument_activation_function_hidden ist entweder eine Klasse oder
        # None (KEINE Instanz). Falls die gewuenschte Aktivierung also nicht
        # None ist, muessen Sie eine Instanz der uebergebenen Aktivierungsfunktions-
        # klasse anlegen.
        #
        # Alle Schichten sollen als Elemente in die self.layers Liste eingefuegt
        # werden. Beachten Sie dabei, dass die Schichten in der richtigen Reihen-
        # fogle eingefuegt werden.

        for dim in n_hidden:
            raise NotImplementedError('Implement me')

        #
        # Fuegen Sie eine letzte vollvernetzte Schicht als Ausgabeschicht
        # hinzu. Beachten Sie auch, das eine Aktivierungsfunktion fuer die
        # Ausgabeschicht erwuenscht sein kann. Fuer diesen Fall sollen Sie
        # zusaetzlich eine Aktivierungsfunktion fuer die Ausagbe einfuegen.
        # Beachten Sie zudem, dass activation_function_output auch eine Klasse
        # ist und KEINE Instanz.

        raise NotImplementedError('Implement me')

    def forward(self, x_input):
        ''' Berechne die Ausgabe des Perceptrons (Forward Pass) '''
        #
        # Berechnen Sie die Ausgabe des MLPs, indem Sie die Eingabe durch alle
        # Schichten propagieren.

        raise NotImplementedError('Implement me')

    def backward(self, top_gradient):
        ''' Berechne die Gradienten der Gewichte und der Eingaben (Backward Pass) '''
        #
        # Berechnen Sie die Gradienten der einzelnen Schichten indem Sie den
        # Fehler pro Schicht jeweils zurueck an die vorhergehende propagieren.
        #
        # Tipp: fuer eine beliebige Liste "a" erhalten Sie die Liste in umge-
        # kehrter Reihenfolge durch "a[::-1]".

        raise NotImplementedError('Implement me')

    def apply_update(self, learning_rate, momentum):
        '''
        Ein Schritt Gradientenabstieg mit Momentum

        Die im backward Aufruf berechneten Gradienten werden verwendet, um die
        Gewichte zu aktualisieren. Dies entspricht einem Schritt im
        Gradientenabstieg.
        '''
        #
        # Wenden Sie die Gradienten in einem Schritt Gradientenabstieg an
        # und veraendern Sie die Gewichte der einzelnen Schichten.
        #
        # Tipp: Verwenden Sie hierfuer die entsprechenden Methodne der
        # Schichten.

        raise NotImplementedError('Implement me')

