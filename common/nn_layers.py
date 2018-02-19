import numpy as np


################ Fully Connected Layer ################
class FullyConnectedLayer(object):
    #
    #
    # Tipp: Sie koennen die Korrektheit der Funktionalitaet dieser Klasse mit
    # Hilfe der Unittests in tests/perceptron_tests.npy ueberpruefen.
    #
    #
    def __init__(self, n_input, n_output):
        '''
        Konstruktor

        :param n_input: Groesse der Eingabedaten (Dimensionen)
        :param n_output: Ausgabegroesse (Dimensionen)
        '''
        #
        # Initialisieren Sie die Gewichtsparameter in dem Sie aus einer
        # Normalverteilung mit Mittelwert 0 und Standardabweichung 0.01 ziehen.
        # Beachten Sie, dass zusaetzliche Gewichte fuer die Biases mit in
        # die Gewichtsmatrix aufgenommen werden muessen.
        # Speichern Sie die Gewichte in der Klassenvariablen self.weights.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.normal.html
        raise NotImplementedError('Implement me')

        #
        # Die folgenden Klassenvariablen werden fuer den Backpropagation
        # Algorithmus benoetigt
        self.x_input = None
        self.gradient_weights = None
        self.gradient_momentum = np.zeros(self.weights.shape)

    def forward(self, x_input):
        '''
         Berechne die Ausgabe der Fully Connected Schicht (Forward Pass)

        :param x_input: Eingabe fuer diese Schicht
        '''
        #
        # Speichern Sie die den Inhalt von x_input in der dafuer vorgesehenen
        # Klassenvariablen self.data.
        # Beachten Sie, dass die gespeicherten Daten noch um eine Dimension
        # erweitert werden muessen, damit die Biases mitgelernt werden.
        #
        # Beachten Sie, dass die Daten row-major vorliegen, dass heisst, dass
        # die Matrix data die folgende Form hat: (n_samples, n_dimensions)
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html
        raise NotImplementedError('Implement me')

        # Berechnen Sie die Linearkombination der von Daten und der Gewichte,
        # also die Ausgabe dieser Schicht, und geben Sie das Ergebnis zurueck
        #
        # Beachten Sie, dass die Daten row-major vorliegen, dass heisst, dass
        # die Matrix x_input die folgende Form hat: (n_samples, n_dimensions)
        raise NotImplementedError('Implement me')

    def backward(self, top_gradient):
        '''
        Backward Pass durch die Fully Connected Schicht (Backward Pass).
        '''
        #
        # Bestimmen Sie den Gradienten der Gewichte und speichern
        # Sie ihn in der Klassenvariablen self.gradient_weights.
        raise NotImplementedError('Implement me')

        #
        # Die Rueckgabe dieser Funktion soll der Gradient in Bezug zur
        # Eingabe sein.
        # Berechnen Sie den Gradient der Eingabe fuer diese Schicht.
        # Dieser Gradient soll in der lokalen Variablen grad_data
        # gespeichert werden.
        raise NotImplementedError('Implement me')

        #
        # Der Gradient hat auch noch die Komponenten fuer die Biases.
        # Diese werden im folgenden einfach abgeschnitten, um die Rueckgabe
        # konsistent zu halten.
        grad_data = grad_data[:, :-1]

        return grad_data

    def apply_update(self, learning_rate, momentum):
        '''
        Ein Schritt Gradientenabstieg mit Momentum

        Die im backward Aufruf berechneten Gradienten werden verwendet, um die
        Gewichte zu aktualisieren. Dies entspricht einem Schritt im
        Gradientenabstieg.
        '''
        # Berechnen Sie den Gradient mit Momentum, das heisst die
        # Linearkombination aus den aktuellen Gradienten der Gewichte und
        # des letzten Gradienten, der in der Variablen self.gradient_momentum
        # gespeichert ist.
        #
        # In welcher Variablen muss der Gradient gespeichert werden?
        #
        # Veraendern Sie anschliessend die Gewichte entsprechend des Gradienten
        # mit Momentum.
        raise NotImplementedError('Implement me')


################ Sigmoid Layer ################
class SigmoidLayer(object):
    #
    #
    # Tipp: Sie koennen die Korrektheit der Funktionalitaet dieser Klasse mit
    # Hilfe der Unittests in tests/mlp_tests.npy ueberpruefen.
    #
    #
    def __init__(self):
        ''' Konstruktur '''
        self.output = None

    def forward(self, x_input):
        '''
         Berechne den Forward Pass fuer die Sigmoid Schicht

        :param x_input: Eingabedaten fuer die Schicht
        '''
        #
        # Implementieren Sie den Forward Pass durch die Sigmoid Schicht.
        # Speichern Sie ausserdem die Ausgabe in der dafuer vorgesehenen
        # Klassenvariablen ab.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
        raise NotImplementedError('Implement me')

    def backward(self, top_gradient):
        '''
         Berechne den Backward Pass fuer die Sigmoid Schicht
          
        :param top_gradient: Der Gradient der hoeher liegenden Schicht
        '''
        #
        # Implementieren Sie den Backward Pass durch die Sigmoid Schicht.
        # Tipp: Verwenden Sie die bereits im Forward Pass berechnete Ausgabe.
        raise NotImplementedError('Implement me')

    def apply_update(self, learning_rate, momentum):
        #
        # Die Sigmoid Schicht hat keine Gewichte, die trainiert werden koennen.
        # Daher entfaellt ein moelgicher Update-Schritt.
        pass


################ ReLU ################
class ReLULayer(object):
    #
    #
    # Tipp: Sie koennen die Korrektheit der Funktionalitaet dieser Klasse mit
    # Hilfe der Unittests in tests/mlp_tests.npy ueberpruefen.
    #
    #
    def __init__(self):
        ''' Konstruktur '''
        self.output = None

    def forward(self, x_input):
        '''
         Berechne den Forward Pass fuer die ReLU Schicht

        :param x_input: Eingabedaten fuer die Schicht
        '''
        #
        # Implementieren Sie den Forward Pass durch die ReLU Schicht.
        # Speichern Sie ausserdem die Ausgabe in der dafuer vorgesehenen
        # Klassenvariablen ab.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html
        raise NotImplementedError('Implement me')

    def backward(self, top_gradient):
        '''
         Berechne den Backward Pass fuer die ReLU Schicht

        :param top_gradient: Der Gradient der hoeher liegenden Schicht
        '''
        #
        # Implementieren Sie den Backward Pass durch die ReLU Schicht.
        raise NotImplementedError('Implement me')

    def apply_update(self, learning_rate, momentum):
        #
        # Die ReLU Schicht hat keine Gewichte, die trainiert werden koennen.
        # Daher entfaellt ein moelgicher Update-Schritt.
        pass
