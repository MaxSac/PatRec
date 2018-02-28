import numpy as np


################ EuclideanLoss ################
class EuclideanLoss(object):
    #
    #
    # Tipp: Sie koennen die Korrektheit der Funktionalitaet dieser Klasse mit
    # Hilfe der Unittests in tests/perceptron_tests.npy ueberpruefen.
    #
    #
    def __init__(self):
        ''' Konstruktor '''
        self.y_label = None
        self.y_pred = None

    def __call__(self, y_pred, y_label):
        return self.loss(y_pred, y_label)

    def loss(self, y_pred, y_label):
        '''
        Berechne den Euclidean Loss

        :param y_pred: Vorhersage des neuronalen Netzes
        :param y_label: Labels, die vorhergesagt werden sollen
        '''
        #
        # Speichern Sie zunaechst die beiden Variablen in den dafuer vorgesehenen
        # Klassenvariablen. Berechnen Sie anschliessend den Euclidean Loss und
        # geben Sie ihn zurueck.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
        self.y_label = y_label
        self.y_pred = y_pred

        diff = self.y_label - self.y_pred
        square = np.square(diff)
        sumed = np.sum(square)
        
        return 0.5*np.sqrt(sumed)
        #raise NotImplementedError('Implement me')

    def gradient(self, scaling_factor=1.0):
        '''
         Berechne den Gradienten der Netzvorhersage

        :param scaling_factor: Skalierungsfaktor fuer den Gradienten
        '''
        #
        # Berechnen Sie den Gradienten des Euklidischen Fehlers bezueglich der
        # Vorhersage des neuronalen Netzes und geben Sie ihn zurueck.
        return scaling_factor * (self.y_label - self.y_pred)
        #raise NotImplementedError('Implement me')


################ BinaryCrossEntropyLoss ################
class BinaryCrossEntropyLoss(object):
    #
    #
    # Tipp: Sie koennen die Korrektheit der Funktionalitaet dieser Klasse mit
    # Hilfe der Unittests in tests/mlp_tests.npy ueberpruefen.
    #
    #
    def __init__(self):
        ''' Konstruktor '''
        self.y_label = None
        self.y_pred = None

    def __call__(self, y_pred, y_label):
        return self.loss(y_pred, y_label)

    def loss(self, y_pred, y_label):
        ''' Berechne den Binary Cross Entropy Loss '''
        #
        # Speichern Sie die Vorhersage des Netzes und die gewuenschten Label
        # in den dafuer vorgesehenen Klassenvariablen. Bestimmen Sie
        # anschliessend den Binary Cross Entropy Loss und geben Sie ihn zurueck.
        #
        # Nuetzliche Funktionen:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
        raise NotImplementedError('Implement me')

    def gradient(self, scaling_factor=1.0):
        '''
         Berechne den Gradienten der Netzvorhersage

        :param scaling_factor: Skalierungsfaktor fuer den Gradienten
        '''
        #
        # Berechnen Sie den Gradienten der binaeren Kreuzentropie bezueglich der
        # Vorhersage des neuronalen Netzes und geben Sie ihn zurueck.
        raise NotImplementedError('Implement me')
