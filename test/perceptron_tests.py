import unittest

import numpy as np

from common.nn_layers import FullyConnectedLayer
from common.nn_loss import EuclideanLoss


class PerceptronTest(unittest.TestCase):


    def testFC(self):
        x_input = np.load('neural_network_data/x_input.npy')
        weights = np.load('neural_network_data/fc_weights.npy')
        exp_output = np.load('neural_network_data/fc_expected_output_forward.npy')
        exp_grad_input = np.load('neural_network_data/fc_expected_grad_input.npy')
        exp_grad_weights = np.load('neural_network_data/fc_expected_grad_weights.npy')
        fc = FullyConnectedLayer(100, 100)
        fc.weights = weights
        output = fc.forward(x_input=x_input)
        grad_input = fc.backward(top_gradient=x_input)
        np.testing.assert_almost_equal(actual=output,
                                       desired=exp_output,
                                       decimal=5,
                                       err_msg='Die Forward Pass Method fuer die Fully Connected Schicht ist fehlerhaft.')
        np.testing.assert_almost_equal(actual=grad_input,
                                       desired=exp_grad_input,
                                       decimal=5,
                                       err_msg='Der Gradient der Fully Connected Schicht bezueglich der Eingabe ist fehlerhaft.')
        np.testing.assert_almost_equal(actual=fc.gradient_weights,
                                       desired=exp_grad_weights,
                                       decimal=5,
                                       err_msg='Der Gradient der Fully Connected Schicht bezueglich der Gewichte ist fehlerhaft.')

    def testEuclideanLoss(self):
        y_pred = np.load('neural_network_data/y_pred.npy')
        y_label = np.load('neural_network_data/y_label.npy')
        exp_grad = np.load('neural_network_data/l2_expected_grad.npy')
        l2loss = EuclideanLoss()
        loss = l2loss.loss(y_pred=y_pred, y_label=y_label)
        grad = l2loss.gradient(scaling_factor=1)
        self.assertAlmostEqual(first=loss,
                               second=832.704727643,
                               places=5,
                               msg='Berechnenung des EuclideanLoss ist fehlerhaft')
        np.testing.assert_almost_equal(actual=grad,
                                       desired=exp_grad,
                                       decimal=5,
                                       err_msg='Der Gradient des Euclidean Loss ist fehlerhaft.')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()