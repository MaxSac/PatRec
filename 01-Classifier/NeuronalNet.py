import numpy as np

class Fully_Connected_Layer(object):
    def __init__(self, n_input, n_output):
        self.n_input = n_input 
        self.n_output = n_output 
        self.weights = np.random.normal(0, 0.01, (n_input+1 ,n_output))

        #for backprop
        self.gradient_weights = None

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
            epochs, learning_rate, momentum):
        self.activation_function= SigmoidLayer()
        self.fc_lay = Fully_Connected_Layer(n_input, n_output)

        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, x_input):
        x_ful_input = self.fc_lay.forward(x_input)
        x_act_input = self.activation_function.forward(x_ful_input)
        return x_act_input

    def backward(self, topgradient=None):
        weights = self.fc_lay.weights[1:,:] # weights ohne Bias
        activat = self.activation_function.backward()

        return weights, activat

    def apply_update(self, learning_rate):
        print(self.fc_lay.gradient_weights)
        self.fc_lay.weights -= learning_rate*self.fc_lay.gradient_weights
            
    def estimate(self,train_samples, train_labels):
        y_pred = self.forward(train_samples)
        self.loss.loss(y_pred, train_labels)
        for x in range(self.epochs):
            self.apply_update(self.learning_rate)

class MultilayerPerceptron:
    def __init__(self, n_input, n_hidden, n_output,
                 activation_function_hidden,
                 activation_function_output,
                 loss,
                 batch_size, epochs, learning_rate, momentum):
                 # init the layers
        self.layers = []
        last_n = n_input
        self.loss = loss 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.epochs = epochs
        for dim in n_hidden:
            self.layers.append(
                    Perceptron(last_n, dim, activation_function_hidden, 
                        loss, batch_size, epochs, learning_rate, momentum)
            )
            last_n = dim
        self.layers.append(
               Perceptron(last_n, n_output, activation_function_output, 
                   loss, batch_size, epochs, learning_rate,momentum)
        )
        
    def forward(self, x_input):
        for percep in self.layers:
            x_input = percep.forward(x_input)
        return x_input


    def backward(self, topgradient=None):
        grad_loss = self.loss.gradient()
        d_activat_Out = self.layers[-1].activation_function.backward()
        new_weights, _ = self.layers[-1].backward()

        print('Grad loss:', grad_loss.shape)
        print(grad_loss)
        print('d_activat_Out:', d_activat_Out.shape)
        print(d_activat_Out)
        topgradient = [grad_loss * d_activat_Out] # local error

        for perc in self.layers[-2::-1]:
#            print('Topgradient: ', topgradient[-1].shape)
#            print(topgradient[-1])

            old_weights = new_weights
            
            print('old_weights: ', old_weights.shape)
            print(old_weights)

            new_weights, d_activat = perc.backward()
            prod = np.dot(topgradient[-1], old_weights) 
            print('Topgradient:', topgradient[-1])
            topgradient.append(d_activat * prod)
            print('Topgradient:', topgradient)
        return topgradient

    def apply_update(self, learning_rate):
        topgradient = self.backward()
        for perc, top in zip(self.layers[-1::-1], topgradient):
            perc.fc_lay.gradient_weights = np.transpose(
                np.dot(np.transpose(top), perc.fc_lay.backward()))
#            print('Weights before:', perc.fc_lay.weights.shape)
#            print(perc.fc_lay.weights)
            perc.apply_update(learning_rate)
#            print('Weights after:', perc.fc_lay.weights.shape)
#            print(perc.fc_lay.weights)

    def estimate(self,train_samples, train_labels):
        for x in range(self.epochs):
            rnd_ch = np.random.choice(
                    train_samples.shape[0], 
                    self.batch_size, 
                    replace=False)
            y_pred = self.forward(train_samples[rnd_ch])
            # print('v_pred', y_pred)
            # print('train_label', train_labels)
            self.loss.loss(y_pred, train_labels[rnd_ch])
            self.apply_update(self.learning_rate)

class EuclideanLoss(object):
    def __init__(self):
        self.y_label = None
        self.y_pred = None

    def loss(self, y_pred, y_label):
        self.y_label = y_label
        self.y_pred = y_pred

        squares = np.square(self.y_label - self.y_pred )
        loss = np.sum(squares, axis=0)/(2*squares.shape[0])
        return loss

    def gradient(self, scaling_factor=1.):
        grad = scaling_factor*(self.y_label - self.y_pred)
        # print('label', self.y_label)
        # print('pred', self.y_pred)
        # print('Gradient loss:', grad)
        return grad

