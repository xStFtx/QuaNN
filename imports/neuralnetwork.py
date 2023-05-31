import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='quaternion_sigmoid'):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation = activation

        for i in range(1, len(layers)):
            weight_shape = (layers[i-1], layers[i])
            self.weights.append(self.quaternion_weight_init(weight_shape))
            self.biases.append(np.zeros((1, layers[i]), dtype=np.complex128))

    def quaternion_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def quaternion_relu(self, x):
        return np.maximum(x, 0)

    def quaternion_tanh(self, x):
        return np.tanh(x)

    def quaternion_qrelu(self, x):
        return x * (x > 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def quaternion_weight_init(self, shape, method='xavier'):
        if method == 'xavier':
            scale = np.sqrt(2 / np.sum(shape))
        elif method == 'he':
            scale = np.sqrt(2 / shape[0])
        else:
            raise ValueError("Unsupported weight initialization method")

        real_part = np.random.normal(0, scale, shape)
        imag_part = np.random.normal(0, scale, shape)
        return real_part + 1j * imag_part

    def forward_pass(self, inputs, training=True):
        activations = [inputs]
        dropout_masks = []

        for i in range(len(self.layers) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            if self.activation == 'quaternion_sigmoid':
                a = self.quaternion_sigmoid(z)
            elif self.activation == 'quaternion_relu':
                a = self.quaternion_relu(z)
            elif self.activation == 'quaternion_tanh':
                a = self.quaternion_tanh(z)
            elif self.activation == 'quaternion_qrelu':
                a = self.quaternion_qrelu(z)
            elif self.activation == 'softmax':
                a = self.softmax(z)
            elif self.activation == 'leaky_relu':
                a = self.leaky_relu(z)
            elif self.activation == 'elu':
                a = self.elu(z)
            else:
                raise ValueError("Unsupported activation function")
            
            if training:
                # Apply dropout regularization
                dropout_mask = np.random.binomial(1, 0.8, size=a.shape) / 0.8
                a *= dropout_mask
                dropout_masks.append(dropout_mask)
            else:
                dropout_masks.append(None)

            activations.append(a)

        return activations, dropout_masks


    def backward_pass(self, inputs, targets, learning_rate, l2_lambda=0):
        activations, dropout_masks = self.forward_pass(inputs, training=True)
        output = activations[-1]
        num_samples = inputs.shape[0]

        if self.activation == 'quaternion_sigmoid':
            delta = (output - targets) * output * (1 - output)
        elif self.activation == 'quaternion_relu':
            delta = (output - targets) * (output > 0)
        elif self.activation == 'quaternion_tanh':
            delta = (output - targets) * (1 - output ** 2)
        elif self.activation == 'quaternion_qrelu':
            delta = (output - targets) * (output > 0)
        elif self.activation == 'softmax':
            delta = output - targets
        elif self.activation == 'leaky_relu':
            delta = (output - targets) * np.where(output > 0, 1, 0.01)
        elif self.activation == 'elu':
            delta = (output - targets) * np.where(output > 0, 1, output + 1)
        else:
            raise ValueError("Unsupported activation function")

        weight_grads = []
        bias_grads = []
        delta_next = delta

        for i in range(len(self.layers) - 2, -1, -1):
            weight_grad = activations[i].T @ delta_next / num_samples
            bias_grad = np.mean(delta_next, axis=0, keepdims=True)

            # Apply L2 regularization
            weight_grad += (l2_lambda / num_samples) * self.weights[i]

            weight_grads.insert(0, weight_grad)
            bias_grads.insert(0, bias_grad)

            delta_next = (delta_next @ self.weights[i].T) * (activations[i] > 0) * dropout_masks[i][:, :activations[i].shape[1]]

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]

    def train(self, inputs, targets, learning_rate, epochs, l2_lambda=0, verbose=True):
        for epoch in range(epochs):
            self.backward_pass(inputs, targets, learning_rate, l2_lambda)

            if verbose and epoch % 100 == 0:
                loss = self.loss(inputs, targets)
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, inputs):
        activations, _ = self.forward_pass(inputs, training=False)
        return activations[-1]

    def loss(self, inputs, targets):
        output = self.predict(inputs)
        return np.mean((output - targets) ** 2)

    def save_model(self, filename):
        model_data = {
            'layers': self.layers,
            'weights': self.weights,
            'biases': self.biases,
            'activation': self.activation
        }
        np.save(filename, model_data)

    @classmethod
    def load_model(cls, filename):
        model_data = np.load(filename, allow_pickle=True).item()
        layers = model_data['layers']
        weights = model_data['weights']
        biases = model_data['biases']
        activation = model_data['activation']
        
        model = cls(layers, activation)
        model.weights = weights
        model.biases = biases
        
        return model
