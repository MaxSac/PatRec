import unittest
import numpy as np
from common.nn_layers import SigmoidLayer, ReLULayer
from common.nn_loss import BinaryCrossEntropyLoss

class MLPTest(unittest.TestCase):


    def testSigmoid(self):
        x_input = np.load('neural_network_data/x_input.npy')
        exp_output = np.load('neural_network_data/sigmoid_expected_output_forward.npy')
        exp_grad_input = np.load('neural_network_data/sigmoid_expected_grad_input.npy')
        sig_layer = SigmoidLayer()
        output = sig_layer.forward(x_input=x_input)
        grad_input = sig_layer.backward(top_gradient=x_input)
        np.testing.assert_almost_equal(actual=output,
                                       desired=exp_output,
                                       decimal=5,
                                       err_msg='Die Forward Pass Method fuer die Sigmoid Schicht ist fehlerhaft.')
        np.testing.assert_almost_equal(actual=grad_input,
                                       desired=exp_grad_input,
                                       decimal=5,
                                       err_msg='Der Gradient der Sigmoid Schicht bezueglich der Eingabe ist fehlerhaft.')

    def testBinaryCrossEntropyLoss(self):
        y_pred = np.load('neural_network_data/y_pred.npy')
        y_label = np.load('neural_network_data/y_label.npy')
        exp_grad = np.load('neural_network_data/bcel_expected_grad.npy')
        bcel = BinaryCrossEntropyLoss()
        loss = bcel.loss(y_pred=y_pred, y_label=y_label)
        grad = bcel.gradient(scaling_factor=1)
        self.assertAlmostEqual(first=loss,
                               second=3072.47638649,
                               places=5,
                               msg='Berechnenung des EuclideanLoss ist fehlerhaft')
        np.testing.assert_almost_equal(actual=grad,
                                       desired=exp_grad,
                                       decimal=5,
                                       err_msg='Der Gradient des Euclidean Loss ist fehlerhaft.')

    def testReLU(self):
        x_input = np.load('neural_network_data/x_input.npy')
        exp_output = np.load('neural_network_data/relu_expected_output_forward.npy')
        exp_grad_input = np.load('neural_network_data/relu_expected_grad_input.npy')
        relu = ReLULayer()
        output = relu.forward(x_input=x_input)
        grad_input = relu.backward(top_gradient=x_input)
        np.testing.assert_almost_equal(actual=output,
                                       desired=exp_output,
                                       decimal=5,
                                       err_msg='Die Forward Pass Method fuer die ReLU Schicht ist fehlerhaft.')
        np.testing.assert_almost_equal(actual=grad_input,
                                       desired=exp_grad_input,
                                       decimal=5,
                                       err_msg='Der Gradient der ReLU Schicht bezueglich der Eingabe ist fehlerhaft.')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()