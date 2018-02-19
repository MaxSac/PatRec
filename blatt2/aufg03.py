from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from common import visualization
from common.data_provider import DataProvider
from common.vector_quantization import Lloyd


def aufg03():
    #
    # Implementieren Sie einen Vektorquantisierer nach Lloyd.
    # Fuer Ihre Implementierung finden Sie einen Klassenrumpf in dem Modul
    # common.vector_quantization
    #
    # Ein anderer Vektorquantisierer ist der k-means-Algorithmus (nach McQueen,
    # siehe z.B. Skript Spracherkennung, S. 53 ff.).
    # Diskutieren Sie die Unterschiede zwischen Lloyd und k-means.
    # Achtung: In der Literatur ist mit k-means in der Regel der Lloyd-Algorithmus
    # gemeint.
    #
    # Welche Codebuchgroesse ist fuer die gegebene Verteilung von Daten geeignet?

    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    train_data = train_data_provider.get_dataset_arr()

    #
    # Waehlen Sie eine geeignete Codebuchgroesse und speichern Sie sie in der
    # lokalen Variablen codebook_size.

    raise NotImplementedError('Implement me')

    #
    # Im Nachfolgenden werden die Daten unabhaengig von der Klassenzugehoerigkeit
    # geclustert.
    lloyd_quantizer = Lloyd()
    codebook = lloyd_quantizer.cluster(train_data, codebook_size)

    #
    # Quantisieren Sie die Trainingsdaten mit dem berechneten Codebuch.
    # Speichern Sie die Clusterindices der jeweiligen Samples in der lokalen
    # Variablen labels.
    #
    # Nuetzliche Funktionen:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html

    raise NotImplementedError('Implement me')

    #
    # Berechnen Sie nun eine Normalverteilung je Cluster und visualisieren Sie diese.
    # Ueberlegen Sie, wie Normalverteilungen definiert sind und wie Sie die noetigen
    # Paramter auf Grundlage der Quantisierung bestimmen koennen.
    # Im Nachfolgenden ist bereits die Farbcodierung der Cluster implementiert.
    # Sie koennen die Variable c als Farbwerte in der visualization.plot_norm_dist_ellipse
    # Funktion verwenden.
    #
    # Nuetzliche Funktionen: np.cov
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
    # Zur Visualisierung: visualization.plot_norm_dist_ellipse
    
    cmap = cm.get_cmap('hsv')
    c = cmap(np.linspace(0, 1, codebook_size))

    raise NotImplementedError('Implement me')

    #
    # Visualisieren Sie die Zugehoerigkeit der einzelnen Samples zu den
    # Clustern. Im Nachfolgenden ist bereits die Farbcodierung der Daten
    # implementiert.
    # Sie koennen die Variable c als Farbwerte in der ax.scatter
    # Funktion verwenden.
    labels_norm = labels / float(codebook_size - 1)
    cmap = cm.get_cmap('hsv')
    c = cmap(labels_norm)
    raise NotImplementedError('Implement me')

    # (optional) Implementieren Sie auch den k-means-Algorithmus nach McQueen.
    # Vergleichen Sie die Laufzeit und die Qualitaet des erzeugten Modells
    # mit dem Lloyd-Algorithmus.

    # (optional) Trainieren Sie das Mischverteilungsmodell mit dem EM-Algorithmus
    # (siehe z.B. Skript Mustererkennung, S. 73 ff.)


if __name__ == '__main__':
    aufg03()
