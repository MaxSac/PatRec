import numpy as np
class Fully_Connected_Layer(object):
    def __init__(self, n_input, n_output):
        self.n_input = n_input 
        self.n_output = n_output 
        self.weights = np.random.normal(0, 0.01, (n_input+1 ,n_output))

    def forward(self, x_input):
        self.data = np.column_stack((np.ones(x_input.shape[0]), x_input))
        return np.dot(self.data, self.weights)

    def backward(self):
        return self.data

class SigmoidLayer(object):
    def __init__(self):
        self.output = None

    def sigmoid(self, node):
        return 0.5*(1+np.tanh(node*0.5))

    def forward(self, x_input):
        self.output = self.sigmoid(x_input)
        return self.output

    def backward(self):
        return self.output*(1-self.output)

class Perceptron:
    def __init__(self, n_input, n_output, activation_function, loss, batch_size,
            epochs, learning_rate):
        self.activation_function= activation_function
        self.fc_lay = Fully_Connected_Layer(n_input, n_output)

        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, x_input):
        x_ful_input = self.fc_lay.forward(x_input)
        x_act_input = self.activation_function.forward(x_ful_input)
        return x_act_input

    def backward(self, topgradient=None):
        if(topgradient==None):
            grad_loss = self.loss.gradient()
            print('grad loss:', grad_loss.shape)
            print(grad_loss)
            grad_acti = self.activation_function.backward()
            print('grad activation', grad_acti.shape)
            print(grad_acti)
            grad_full = self.fc_lay.backward()
            print('grad fully', grad_full.shape)
            print(grad_full)
            print('-------------------------------')
            print('delta_out')
            delta_out = grad_acti*grad_loss
            print(delta_out)
            print('delta_out * input fc')
            delta_weights = delta_out*np.transpose(self.fc_lay.data)
            print(delta_weights)


    def estimate(self,train_samples, train_labels):
        y_pred = self.forward(train_samples)
        self.loss.loss(y_pred, train_labels)
        self.backward()

class EuclideanLoss(object):
    def __init__(self):
        self.y_label = None
        self.y_pred = None

    def loss(self, y_pred, y_label):
        self.y_label = y_label
        self.y_pred = y_pred

        squares = np.square(self.y_pred - self.y_label)
        loss = np.sum(squares, axis=0)/(2*squares.shape[0])
#        print('loss: ---------------------')
#        print(loss)
        return loss

    def gradient(self, scaling_factor=1.):
        grad = (self.y_pred - self.y_label)
#        print('grad: ---------------------')
#        print(grad)
        return grad

