import numpy as np
from scipy.spatial.transform import Rotation

class QuaternionNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.weights_ih = np.random.uniform(-1, 1, (hidden_dim, input_dim, 4))
        self.weights_ho = np.random.uniform(-1, 1, (output_dim, hidden_dim, 4))
        self.biases_h = np.zeros((hidden_dim, 4))
        self.biases_o = np.zeros((output_dim, 4))
        
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_w_ih = np.zeros_like(self.weights_ih)
        self.m_w_ho = np.zeros_like(self.weights_ho)
        self.m_b_h = np.zeros_like(self.biases_h)
        self.m_b_o = np.zeros_like(self.biases_o)
        self.v_w_ih = np.zeros_like(self.weights_ih)
        self.v_w_ho = np.zeros_like(self.weights_ho)
        self.v_b_h = np.zeros_like(self.biases_h)
        self.v_b_o = np.zeros_like(self.biases_o)
        self.t = 0
    
    def forward(self, input_quaternion):
        hidden_quaternions = np.zeros((self.hidden_dim, 4))
        output_quaternions = np.zeros((self.output_dim, 4))

        # Quaternion activation function
        def activate_quaternion(q):
            return q / np.linalg.norm(q)

        # Input to hidden layer
        for i in range(self.hidden_dim):
            hidden_quaternion = np.zeros(4)
            for j in range(self.input_dim):
                hidden_quaternion += np.quaternion(*self.weights_ih[i][j]) * input_quaternion[j]
            hidden_quaternions[i] = activate_quaternion(hidden_quaternion + self.biases_h[i])

        # Hidden to output layer
        for i in range(self.output_dim):
            output_quaternion = np.zeros(4)
            for j in range(self.hidden_dim):
                output_quaternion += np.quaternion(*self.weights_ho[i][j]) * hidden_quaternions[j]
            output_quaternions[i] = activate_quaternion(output_quaternion + self.biases_o[i])

        return output_quaternions
    
    def backward(self, input_quaternion, target_quaternion, learning_rate):
        output_quaternions = self.forward(input_quaternion)
        
        # Calculate gradients
        output_errors = 2 * (output_quaternions - target_quaternion)
        hidden_errors = np.zeros((self.hidden_dim, 4))
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                hidden_errors[j] += np.quaternion(*self.weights_ho[i][j]) * output_errors[i]
                
        # Update weights and biases using Adam optimizer
        gradient_w_ih = np.zeros_like(self.weights_ih)
        gradient_w_ho = np.zeros_like(self.weights_ho)
        gradient_b_h = np.zeros_like(self.biases_h)
        gradient_b_o = np.zeros_like(self.biases_o)
        
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                gradient_w_ih[i][j] = np.quaternion(*hidden_errors[i]) * input_quaternion[j]
                
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                gradient_w_ho[i][j] = np.quaternion(*output_errors[i]) * np.conj(np.quaternion(*input_quaternion[j]))
                
        gradient_b_h = hidden_errors
        gradient_b_o = output_errors
        
        self.t += 1
        self.m_w_ih = self.beta1 * self.m_w_ih + (1 - self.beta1) * gradient_w_ih
        self.m_w_ho = self.beta1 * self.m_w_ho + (1 - self.beta1) * gradient_w_ho
        self.m_b_h = self.beta1 * self.m_b_h + (1 - self.beta1) * gradient_b_h
        self.m_b_o = self.beta1 * self.m_b_o + (1 - self.beta1) * gradient_b_o
        self.v_w_ih = self.beta2 * self.v_w_ih + (1 - self.beta2) * (gradient_w_ih ** 2)
        self.v_w_ho = self.beta2 * self.v_w_ho + (1 - self.beta2) * (gradient_w_ho ** 2)
        self.v_b_h = self.beta2 * self.v_b_h + (1 - self.beta2) * (gradient_b_h ** 2)
        self.v_b_o = self.beta2 * self.v_b_o + (1 - self.beta2) * (gradient_b_o ** 2)
        
        m_w_ih_hat = self.m_w_ih / (1 - self.beta1 ** self.t)
        m_w_ho_hat = self.m_w_ho / (1 - self.beta1 ** self.t)
        m_b_h_hat = self.m_b_h / (1 - self.beta1 ** self.t)
        m_b_o_hat = self.m_b_o / (1 - self.beta1 ** self.t)
        v_w_ih_hat = self.v_w_ih / (1 - self.beta2 ** self.t)
        v_w_ho_hat = self.v_w_ho / (1 - self.beta2 ** self.t)
        v_b_h_hat = self.v_b_h / (1 - self.beta2 ** self.t)
        v_b_o_hat = self.v_b_o / (1 - self.beta2 ** self.t)
        
        self.weights_ih -= learning_rate * m_w_ih_hat / (np.sqrt(v_w_ih_hat) + self.epsilon)
        self.weights_ho -= learning_rate * m_w_ho_hat / (np.sqrt(v_w_ho_hat) + self.epsilon)
        self.biases_h -= learning_rate * m_b_h_hat / (np.sqrt(v_b_h_hat) + self.epsilon)
        self.biases_o -= learning_rate * m_b_o_hat / (np.sqrt(v_b_o_hat) + self.epsilon)
        
    def train(self, input_quaternions, target_quaternions, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(input_quaternions)):
                input_quaternion = input_quaternions[i]
                target_quaternion = target_quaternions[i]
                self.backward(input_quaternion, target_quaternion, learning_rate)
                loss = np.linalg.norm(self.forward(input_quaternion) - target_quaternion)
                total_loss += loss
            avg_loss = total_loss / len(input_quaternions)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Example usage
input_dim = 2
hidden_dim = 3
output_dim = 1

# Generate random input and target quaternions
input_quaternions = [np.quaternion(*Rotation.random().as_quat()) for _ in range(100)]
target_quaternions = [np.quaternion(*Rotation.random().as_quat()) for _ in range(100)]

# Create the quaternion neural network
nn = QuaternionNeuralNetwork(input_dim, hidden_dim, output_dim)

# Train the network
nn.train(input_quaternions, target_quaternions, epochs=100, learning_rate=0.01)
