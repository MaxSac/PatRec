import numpy as np
import random
from scipy.spatial.distance import cdist


class Lloyd(object):

    def cluster(self, samples, codebook_size, prune_codebook=False):
        '''Partitioniert Beispieldaten in gegebene Anzahl von Clustern.

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
            codebook_size: Anzahl von Komponenten im Codebuch
            prune_codebook: Boolsches Flag, welches angibt, ob das Codebuch
                bereinigt werden soll. Die Bereinigung erfolgt auf Grundlage
                einer Heuristik, die die Anzahl der, der Cluster Komponente
                zugewiesenen, Beispieldaten beruecksichtigt.
                Optional, default=False

        Returns:
            codebook: ndarry mit codebook_size Codebuch Vektoren,
                zeilenweise, shape=(codebook_size,t)

        mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
        '''
        
        # Bestimmen Sie in jeder Iteration den Quantisierungsfehler und brechen Sie
        # das iterative Verfahren ab, wenn der Quantisierungsfehler konvergiert
        # (die Aenderung einen sehr kleinen Schwellwert unterschreitet).
        # Nuetzliche Funktionen: scipy.distance.cdist, np.mean, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Fuer die Initialisierung mit zufaelligen Punkten: np.random.permutation
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html
        
        # -----------------Zufaelliger wert zwischen min und max ---------------  
        # codebook = np.array([[],[]])
        # for x in range(codebook_size):
        #     codebook = np.append(codebook ,[ 
        #         [np.random.uniform(min(samples[:,0]),max(samples[:,0]))],
        #         [np.random.uniform( min(samples[:,0]),max(samples[:,0]))]
        #         ], axis=1)
        # codebook = np.transpose(codebook)

        codebook = []

        for i in range(codebook_size):
            codebook.append(random.choice(samples))
        codebook = np.array(codebook)
    
        error = [0]
        while(True):
            dist = cdist(samples, codebook)
            cluster_number = np.argsort(dist)
            error_class = 0
            for x in  cluster_number[0,:]:
                class_sample = samples[cluster_number[:,0]==x]
                new_cluster = np.mean(class_sample, axis=0)
                codebook[x] =  new_cluster
                error_class += np.sum(cdist(class_sample, [new_cluster]))
            error.append(error_class)
            if(abs(error[-2]-error[-1])/error[-1] <= 0.01):
                break 
         
        return codebook
        
        # raise NotImplementedError('Implement me')
