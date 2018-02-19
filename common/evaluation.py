import numpy as np
from _collections import defaultdict


class CrossValidation(object):

    def __init__(self, samples, labels, n_folds):
        '''Initialisiert die Kreuzvalidierung

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
            labels: ndarray mit Labels fuer Beispieldaten, shape=(d,)
            n_folds: Anzahl Folds ueber die die Kreuzvalidierung durchgefuehrt
                werden soll.

        mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
        '''
        self.samples = samples
        self.labels = labels
        self.n_folds = n_folds

    def validate(self, classifier):
        '''Fuert die Kreuzvalidierung mit dem Klassifikator 'classifier' durch.

        Params:
            classifier: Objekt, das folgende Methoden implementiert (siehe oben)
                estimate(train_samples, train_labels)
                classify(test_samples) --> test_labels

        Returns:
            crossval_overall_result: Erkennungsergebnis der gesamten Kreuzvalidierung
                (ueber alle Folds)
            crossval_class_results: Liste von Tuple (category, result) die klassenweise
                Erkennungsergebnisse der Kreuzvalidierung enthaelt.
        '''
        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.n_folds):
            train_samples, train_labels, test_samples, test_labels = self.samples_fold(fold_index)
            classifier.estimate(train_samples, train_labels)
            estimated_test_labels = classifier.classify(test_samples)
            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results

    @staticmethod
    def crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result

    def samples_fold(self, fold_index):
        '''Berechnet eine Aufteilung der Daten in Training und Test

        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.

        Returns:
            train_samples: ndarray mit Trainingsdaten, shape=(d_train,t)
            train_label: ndarray mit Trainingslabels, shape=(d_train,t)
            test_samples: ndarray mit Testdaten, shape=(d_test,t)
            test_label: ndarray mit Testlabels, shape=(d_test,t)

        mit d_{train,test} Beispieldaten und t dimensionalen Merkmalsvektoren.
        '''
        n_samples = self.samples.shape[0]
        test_indices = range(fold_index, n_samples, self.n_folds)
        train_indices = [train_index for train_index in range(n_samples)
                             if train_index not in test_indices]

        test_samples = self.samples[test_indices, :]
        test_labels = self.labels[test_indices]
        train_samples = self.samples[train_indices, :]
        train_labels = self.labels[train_indices]

        return train_samples, train_labels, test_samples, test_labels


class ClassificationEvaluator(object):

    def __init__(self, estimated_labels, groundtruth_labels):
        '''Initialisiert den Evaluator fuer ein Klassifikationsergebnis
        auf Testdaten.

        Params:
            estimated_labels: ndarray (1-Dimensional) mit durch den Klassifikator
                bestimmten Labels (N Komponenten).
            groundtruth_labels: ndarray (1-Dimensional) mit den tatsaechlichen
                Labels (N Komponenten).
        '''
        self.estimated_labels = estimated_labels
        self.groundtruth_labels = groundtruth_labels
        self.binary_result_mat = groundtruth_labels == estimated_labels

    def error_rate(self, mask=None):
        '''Bestimmt die Fehlerrate auf den Testdaten.

        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        '''
        if mask is None:
            mask = np.ones_like(self.binary_result_mat, dtype=bool)
        masked_binary_result_mat = self.binary_result_mat[mask]
        n_samples = len(masked_binary_result_mat)
        n_correct = masked_binary_result_mat.sum()
        n_wrong = n_samples - n_correct
        error_rate = n_wrong / float(n_samples)
        error_rate *= 100
        return error_rate, n_wrong, n_samples

    def category_error_rates(self):
        '''Berechnet klassenspezifische Fehlerraten

        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        '''
        category_list = sorted(set(self.groundtruth_labels.ravel()))
        cat_n_err_list = []
        for category in category_list:
            category_mask = self.groundtruth_labels == category
            err, n_wrong, n_samples = self.error_rate(category_mask)
            cat_n_err_list.append((category, err, n_wrong, n_samples))

        return cat_n_err_list

