from imports.neuralnetwork import NeuralNetwork
from imports.banach import BanachSpace
from imports.fourier import Fourier
import numpy as np

def main():
    # Example training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Create a neural network with three hidden layers
    layers = [2, 4, 4, 4, 1]
    nn = NeuralNetwork(layers, activation='quaternion_sigmoid')

    while True:
        print("\n1. Train the model")
        print("2. Load a pre-trained model")
        print("3. Fourier Banach")
        print("4. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            # Train the neural network
            learning_rate = 0.1
            epochs = 10000
            nn.train(inputs, targets, learning_rate, epochs)

            # Save the trained model
            nn.save_model("trained_model.npy")
            print("Model training complete. Trained model saved as 'trained_model.npy'.")

            # Predict using the trained model
            predictions = nn.predict(inputs)
            print("\nPredictions:")
            print(predictions)

        elif choice == '2':
            # Load a pre-trained model
            try:
                loaded_model = NeuralNetwork.load_model("trained_model.npy")
                print("Pre-trained model loaded successfully.")

                # Predict using the loaded model
                predictions = loaded_model.predict(inputs)
                print("\nPredictions:")
                print(predictions)

            except FileNotFoundError:
                print("No pre-trained model found. Please train the model first.")
        elif choice == '3':
            # Fourier and banach space
            # Create a signal and its corresponding Fourier class instance
            signal = [1, 2, 3, 4, 3, 2, 1]
            fourier = Fourier(signal)

            # Perform Fourier transform
            x = np.arange(len(signal))
            freq, spectrum = fourier.compute_fourier_transform(x, signal)

            # Analyze the spectrum
            analyzed_freq, analyzed_coefficients = fourier.analyze_fourier_coefficients(freq, spectrum, 5)

            # Print the results
            print("Signal: ", signal)
            print("Frequency: ", freq)
            print("Fourier Spectrum: ", spectrum)
            print("Analyzed Frequencies: ", analyzed_freq)
            print("Analyzed Coefficients: ", analyzed_coefficients)

        elif choice == '4':
            print("Exiting the program...")
            break

        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == '__main__':
    main()
