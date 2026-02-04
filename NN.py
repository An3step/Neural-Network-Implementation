import numpy as np

import numpy as np

class Operation():
    '''
    Базовый класс операций
    '''
    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:
        
        self.input_ = input_

        self.output_ = self._output()
        
        return self.output_

    def backward(self, output_diff: np.ndarray) -> np.ndarray:

        assert output_diff.shape[::-1] == self.output_.shape, f"Output_diff_shape = {output_diff.shape}, Output_shape = {self.output_.shape}"

        return self._input_diff(output_diff)

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_diff(self, output_diff_):
        raise NotImplementedError()

class ParamOperation(Operation):

    def __init__(self, params):
        self.params = params

    def backward(self, output_diff: np.ndarray) -> np.ndarray:

        assert output_diff.shape[::-1] == self.output_.shape, f"Output_diff_shape = {output_diff.shape}, Output_shape = {self.output_.shape}"
        self.params_grad = self._params_grad(output_diff)
        assert self.params_grad.shape == self.params.shape, f"Params_grad_shape = {self.params_grad.shape}, Params_shape = {self.params.shape}"
        return self._input_diff(output_diff)

    def _params_grad(self, output_diff_: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_params(self):
        return self.params

    def get_params_grad(self):
        return self.params_grad

class WeightMultiply(ParamOperation):
    
    def _output(self):
        assert self.params.shape[0] == self.input_.shape[1], f"Params_shape = {self.params.shape}, Input_shape = {self.input_.shape}"
        return self.input_ @ self.params

    def _input_diff(self, output_diff_: np.ndarray):
        return self.params @ output_diff_

    def _params_grad(self, output_diff_: np.ndarray):
        return (output_diff_ @ self.input_).T

class BiasAdd(ParamOperation):

    def __init__(self, B : np.ndarray):
        super().__init__(B)
    
    def _output(self):
        return self.input_ + self.params

    def _input_diff(self, output_diff_: np.ndarray):
        return output_diff_

    def _params_grad(self, output_diff_: np.ndarray):
        return np.sum(output_diff_.T, axis=0, keepdims=True)

class Sigmoid(Operation):

    def _sigmoid(self, input_ : np.ndarray):
        return 1.0 / (1.0 + np.exp(np.clip(-input_, -50, 50)))

    def _output(self):
        return self._sigmoid(self.input_)

    def _input_diff(self, output_diff_):
        input_grad = (self.output_ * (1 - self.output_)).T * output_diff_
        return input_grad

from typing import List

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
    
    def __init__(self, neurons, activation = Sigmoid):
        super().__init__(neurons)
        self.activation = activation

    def setup_layer(self, shape):

        W = np.random.randn(shape[1], self.neurons)
        B = np.random.randn(1, self.neurons)
        if self.activation is not None:
            self.operations = [WeightMultiply(W),
                              BiasAdd(B),
                              self.activation()]
        else:
            self.operations = [WeightMultiply(W),
                              BiasAdd(B)]

class Loss():

    def __init__(self):
        pass

    def forward(self, target, prediction):
        self.target = target
        self.prediction = prediction
        assert self.target.shape == self.prediction.shape, f"Target_shape = {self.target.shape}, Prediction_shape = {self.prediction.shape}"
        self.loss_value = self._output()
        return self.loss_value

    def backward(self):
        self.prediction_diff_ = self.prediction_diff()
        assert self.prediction_diff_.shape[::-1] == self.prediction.shape
        return self.prediction_diff_

    def _output(self):
        raise NotImplementedError()

    def prediction_diff(self):
        raise NotImplementedError()

class MSE(Loss):
    
    def _output(self):
        return np.mean((self.target.reshape(self.prediction.shape) - self.prediction) ** 2)

    def prediction_diff(self):
        if self.prediction.ndim == 1:
            return 2 / (self.prediction.shape[0] * self.prediction.shape[1]) * (self.prediction - self.target.reshape(self.prediction.shape)).reshape(-1, 1)
        else:
            return 2 / (self.prediction.shape[0] * self.prediction.shape[1]) * (self.prediction - self.target.reshape(self.prediction.shape)).T

from typing import List

class NeuralNetwork():
    
    def __init__(self, layers: List[Layer | Dense]):
        self.layers = layers
        
    def forward(self, x_batch):

        self.x_output = x_batch
        for layer in self.layers:
            self.x_output = layer.forward(self.x_output)
        return self.x_output

    def backward(self, loss_grad):

        self.loss_grad = loss_grad

        for layer in self.layers[::-1]:

            self.loss_grad = layer.backward(self.loss_grad)

        return None

    def get_params(self):
        return map(lambda layer: layer.get_params(), self.layers)

    def get_params_grad(self):
        return map(lambda layer: layer.get_params_grad(), self.layers)

class Optimizer():

    def __init__(self, learning_rate = 0.01):

        self.lr = learning_rate

    def step(self):
        raise NotImplementedError()

class SGD(Optimizer):

    def step(self):
        for layer_params, layer_params_grad in zip(self.net.get_params(), self.net.get_params_grad()): # type: ignore
            for params, params_grad in zip(layer_params, layer_params_grad):
                params -= self.lr * params_grad

class Trainer:
    def __init__(self, loss: Loss, Net: NeuralNetwork, optimizer : Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.Net = Net
        setattr(self.optimizer, 'net', self.Net)

    def get_batch(self, x_train, y_train, batch_size):
        N = x_train.shape[0]
        for i in range(0, N, batch_size):
            yield x_train[i:i + batch_size], y_train[i:i + batch_size]
    
    def score(self, X):
        return self.Net.forward(X)

    def fit(self, x_train, y_train, epochs = 100, verbose=True, batch_size = None):

        np.random.seed(10)

        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]

        if batch_size is None:
            batch_size = x_train.shape[0] // 5

        for e in range(epochs):
            all_loss = 0
            for x_batch, y_batch in self.get_batch(x_train, y_train, batch_size):
                pred = self.Net.forward(x_batch)
                loss = self.loss.forward(y_batch, pred)
                all_loss += loss
                loss_grad = self.loss.backward()
                self.Net.backward(loss_grad)
                self.optimizer.step()

            if verbose:
                print(f'Epoch {e + 1}: loss = {all_loss / x_train.shape[0]}')

