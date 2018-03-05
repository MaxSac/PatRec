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
            epochs, learning_rate):
        self.activation_function= SigmoidLayer()
        self.fc_lay = Fully_Connected_Layer(n_input, n_output)

        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, x_input):
        x_ful_input = self.fc_lay.forward(x_input)
        x_act_input = self.activation_function.forward(x_ful_input)
        return x_act_input

    def backward(self, topgradient):
        weights = self.fc_lay.weights[1:,:] # weights ohne Bios
        activat = self.activation_function.backward()
        
        self.fc_lay.gradient_weights = topgradient

        delta = topgradient*np.transpose(activat)

        topgradient = np.dot(weights, delta)

        return topgradient

    def apply_update(self, learning_rate):
        delta_weight = np.dot(self.fc_lay.gradient_weights,
                self.fc_lay.backward())
        self.fc_lay.weights += learning_rate*np.transpose(delta_weight)
            
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
        for dim in n_hidden:
            self.layers.append(
                    Perceptron(last_n, dim, activation_function_hidden, 
                        loss, batch_size, epochs, learning_rate)
            )
            last_n = dim
        self.layers.append(
               Perceptron(last_n, n_output, activation_function_output, 
                   loss, batch_size, epochs, learning_rate)
        )
        
    def forward(self, x_input):
        for percep in self.layers:
            x_input = percep.forward(x_input)
        return x_input


    def backward(self, topgradient=None):
        grad_loss = self.loss.gradient()
        grad_acti = self.layers[-1].activation_function.backward()
        weights_without_bios = self.layers[-1].fc_lay.backward()[:,1:]

        loc_error = grad_loss # lokaler Fehler 
        topgradient = np.transpose(loc_error)

        for perc in self.layers[-1::-1]:
           topgradient = perc.backward(topgradient)

    def apply_update(self, learning_rate):
        self.backward()
        for perc in self.layers[::-1]:
            perc.apply_update(learning_rate)

    def estimate(self,train_samples, train_labels):
        for x in range(200):
            rnd_ch = np.random.choice(
                    train_samples.shape[0], 
                    self.batch_size, 
                    replace=False)
            y_pred_b = self.forward(train_samples[rnd_ch])
            self.loss.loss(y_pred_b, train_labels[rnd_ch])
            self.apply_update(self.learning_rate)
            y_pred_a = self.forward(train_samples[rnd_ch])
            print('Value before:', np.round(y_pred_b,2) , ' after: ',
                    np.round(y_pred_a,2), 'truelabel',
                    train_labels[rnd_ch])
        y_pred = self.forward(train_samples)

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
        return grad

