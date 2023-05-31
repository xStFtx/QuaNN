from imports.neuralnetwork import NeuralNetwork
import numpy as np

def main():
    # Example training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Create a neural network with three hidden layers
    layers = [2, 4, 4, 4, 1]
    nn = NeuralNetwork(layers, activation='quaternion_sigmoid')

    # Train the neural network
    learning_rate = 0.1
    epochs = 10000
    nn.train(inputs, targets, learning_rate, epochs)

    # Predict using the trained neural network
    predictions = nn.predict(inputs)
    print("Predictions:")
    print(predictions)


if __name__ == '__main__':
    main()
