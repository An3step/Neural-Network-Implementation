from typing import List
from NN import Operation, ParamOperation, WeightMultiply,\
      BiasAdd, Sigmoid, Loss, MSE, Optimizer, SGD

import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(np.clip(x, -50, 50))
    return exp_x / np.sum(exp_x)

def propab_to_labels(X_pred):
    return np.argmax(X_pred, axis=1)

def accuracy_trainer(trainer, X_test, y_test):
    return (accuracy_score(np.argmax(y_test, axis=1), propab_to_labels(trainer.score(X_test))))

class Dropout(Operation):
    def __init__(self, keep_prob = 0.7):
        super().__init__()
        self.keep_prob = keep_prob
        self.inference = False

    def train(self):
        self.inference = False
    
    def eval(self):
        self.inference = True

    def _output(self):
        if self.inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
            return self.input_ * self.mask
    
    def _input_diff(self, output_diff_):
        return output_diff_ * self.mask.reshape(output_diff_.shape)

class IDSGD(SGD):
    def __init__(self, lr=0.01, momentum=0.9, decay_type='exponential'):
        self.lr = lr
        self.momentum = momentum
        self.First = True
        self.decay_type = decay_type
        if self.decay_type:
            self.start_lr = 1.5 * lr
            self.final_lr = 0.5 * lr
        else:
            self.start_lr = lr
            self.final_lr = lr
        self.lr = self.start_lr
        
    
    def _setup_decay(self, epoch_count):
        if self.decay_type == 'linear':
            self.decay_per_epoch = (self.start_lr - self.final_lr) / epoch_count
        else:
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1 / (epoch_count))
    
    def _decay(self):
        if self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch
        else:
            self.lr *= self.decay_per_epoch
        if self.lr < 0:
            print("LR < 0")
            return -1
        # print("Changed", self.lr)
        return None

    def step(self):
        if self.First:
            self.prev_grads = [[np.zeros_like(oper) for oper in layer] for layer in self.net.get_params()] # type: ignore
            self.First = False
        for layer_params, layer_params_grad, prev_Layer_grad in zip(self.net.get_params(), self.net.get_params_grad(), self.prev_grads): # type: ignore
            for params, params_grad, prev_grad in zip(layer_params, layer_params_grad, prev_Layer_grad):
                grad = self.lr * params_grad - self.momentum*prev_grad
                params -= grad
                prev_grad[:] = grad

class Layer:
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.operations: List[Operation] = []
        self.First = True

    def setup_layer(self, shape):
        raise NotImplementedError()

    def forward(self, input_: np.ndarray):
        self.input_ = input_
        if self.First:
            self.setup_layer(self.input_.shape)
            self.First = False
        self._output = input_
        for operation in self.operations:
            self._output = operation.forward(self._output)
        
        return self._output

    def train(self):
        if isinstance(self.operations[-1], Dropout):
            self.operations[-1].train()
    
    def eval(self):
        if isinstance(self.operations[-1], Dropout):
            self.operations[-1].eval()

    def backward(self, output_diff):

        self.input_grad = output_diff

        for operation in self.operations[::-1]:
            self.input_diff = operation.backward(self.input_grad)
        return self.input_diff

    def get_params(self):
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                yield operation.get_params()

    def get_params_grad(self):
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                yield operation.get_params_grad()

class Dense(Layer):
    
    def __init__(self, neurons, activation = Sigmoid, init='glorot', dropout=1):
        super().__init__(neurons)
        self.activation = activation
        self.init = init
        self.dropout = dropout

    def setup_layer(self, shape):
        if self.init == 'standart':
            W = np.random.randn(shape[1], self.neurons)
            B = np.random.randn(1, self.neurons)
        else:
            W = np.random.randn(shape[1], self.neurons) * (2 / (shape[1] + self.neurons))
            B = np.random.randn(1, self.neurons) * (2 / (shape[1] + self.neurons))
        if self.activation is not None:
            self.operations = [WeightMultiply(W),
                              BiasAdd(B),
                              self.activation()]
        else:
            self.operations = [WeightMultiply(W),
                              BiasAdd(B)]
        if self.dropout < 1:
            self.operations.append(Dropout(self.dropout))

class NeuralNetwork():
    
    def __init__(self, layers: List[Layer | Dense]):
        self.layers = layers
        
    def forward(self, x_batch):

        self.x_output = x_batch
        for layer in self.layers:
            self.x_output = layer.forward(self.x_output)
        return self.x_output

    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()

    def backward(self, loss_grad):

        self.loss_grad = loss_grad

        for layer in self.layers[::-1]:

            self.loss_grad = layer.backward(self.loss_grad)

        return None

    def get_params(self):
        return map(lambda layer: layer.get_params(), self.layers)

    def get_params_grad(self):
        return map(lambda layer: layer.get_params_grad(), self.layers)

class Trainer:
    def __init__(self, loss: Loss, Net: NeuralNetwork, optimizer : Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.Net = Net
        setattr(self.optimizer, 'net', self.Net)
        self.inference = False

    def get_batch(self, x_train, y_train, batch_size):
        N = x_train.shape[0]
        for i in range(0, N, batch_size):
            yield x_train[i:i + batch_size], y_train[i:i + batch_size]
    
    def train(self):
        self.Net.train()
    
    def eval(self):
        self.Net.eval()

    def score(self, X):
        return self.Net.forward(X)

    def fit(self, x_train, y_train, epochs = 100, verbose=True, batch_size = None):

        np.random.seed(10)
        # self.train()
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]

        if batch_size is None:
            batch_size = x_train.shape[0] // 5

        if isinstance(self.optimizer, IDSGD) or isinstance(self.optimizer, DecaySGD):
            # print("NOT ERROR")
            self.optimizer._setup_decay(epochs)
        # else:
            # print("ERROR?")
        for e in range(epochs):
            all_loss = 0
            for x_batch, y_batch in self.get_batch(x_train, y_train, batch_size):
                pred = self.Net.forward(x_batch)
                loss = self.loss.forward(y_batch, pred)
                all_loss += loss
                loss_grad = self.loss.backward()
                self.Net.backward(loss_grad)
                self.optimizer.step()
            if isinstance(self.optimizer, IDSGD) or isinstance(self.optimizer, DecaySGD):
                r = self.optimizer._decay()
                if r:
                    print(f"EPOCH {e}, {r}")
                    c = input()
            if verbose:
                print(f'Epoch {e + 1}: loss = {all_loss / x_train.shape[0]}, lr = {self.optimizer.lr}')

class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self):
        self.epsilon = 1e-7
    def _output(self):
        softmax_pred = np.array([softmax(row) for row in self.prediction])
        self.softmax = np.clip(softmax_pred, self.epsilon, 1 - self.epsilon)
        return np.sum(-self.target*np.log(self.softmax))
    def prediction_diff(self):
        return (self.softmax - self.target).T

class Tanh(Operation):
    def _output(self):
        clipped_input = np.clip(self.input_, -50, 50)
        exp_2x = np.exp(2 * clipped_input)
        return (exp_2x - 1) / (exp_2x + 1)
    
    def _input_diff(self, output_diff_):
        return (output_diff_ * (1 - np.square(self.output_)).T)

class Impulse_SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.First = True
        
    def step(self):
        if self.First:
            self.prev_grads = [[np.zeros_like(oper) for oper in layer] for layer in self.net.get_params()] # type: ignore
            self.First = False
        for layer_params, layer_params_grad, prev_Layer_grad in zip(self.net.get_params(), self.net.get_params_grad(), self.prev_grads): # type: ignore
            for params, params_grad, prev_grad in zip(layer_params, layer_params_grad, prev_Layer_grad):
                grad = self.lr * params_grad - self.momentum*prev_grad
                params -= grad
                prev_grad[:] = grad

class DecaySGD(SGD):
    def __init__(self, start_lr = 0.1, final_lr = 0.01, decay_type='exponential'):
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.lr = self.start_lr
    
    def _setup_decay(self, epoch_count):
        if self.decay_type == 'linear':
            self.decay_per_epoch = (self.start_lr - self.final_lr) / epoch_count
        else:
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1 / (epoch_count))
    
    def _decay(self):
        if self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch
        else:
            self.lr *= self.decay_per_epoch

# class DecayTrainer(Trainer):

#     def fit(self, x_train, y_train, epochs = 100, verbose=True, batch_size = None):

#         np.random.seed(10)

#         indices = np.random.permutation(x_train.shape[0])
#         x_train = x_train[indices]
#         y_train = y_train[indices]

#         if batch_size is None:
#             batch_size = x_train.shape[0] // 5
        
#         if isinstance(self.optimizer, DecaySGD):
#             self.optimizer._setup_decay(epochs)

#         for e in range(epochs):
#             all_loss = 0
#             for x_batch, y_batch in self.get_batch(x_train, y_train, batch_size):
#                 pred = self.Net.forward(x_batch)
#                 loss = self.loss.forward(y_batch, pred)
#                 all_loss += loss
#                 loss_grad = self.loss.backward()
#                 self.Net.backward(loss_grad)
#                 self.optimizer.step()
#             if isinstance(self.optimizer, DecaySGD):
#                 self.optimizer._decay()
#             if verbose:
#                 print(f'Epoch {e + 1}: loss = {all_loss / x_train.shape[0]}, lr = {self.optimizer.lr}')






# class GlorotDense(Dense):

#     def setup_layer(self, shape):
#         W = np.random.randn(shape[1], self.neurons) * (2 / (shape[1] + self.neurons))
#         B = np.random.randn(1, self.neurons) * (2 / (shape[1] + self.neurons))
#         if self.activation is not None:
#             self.operations = [WeightMultiply(W),
#                               BiasAdd(B),
#                               self.activation()]
#         else:
#             self.operations = [WeightMultiply(W),
#                               BiasAdd(B)]