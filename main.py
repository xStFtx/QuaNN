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

    # Save the trained model
    nn.save_model("trained_model.npy")

    # Load the trained model
    loaded_model = NeuralNetwork.load_model("trained_model.npy")

    # Predict using the loaded model
    predictions = loaded_model.predict(inputs)
    print("Predictions:")
    print(predictions)


if __name__ == '__main__':
    main()
