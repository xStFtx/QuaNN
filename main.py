import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from imports.neuralnetwork import NeuralNetwork
from imports.fourier import Fourier
from imports.banach import BanachSpace
from imports.hilbert import HilbertSpace
from imports.quaternion import Quaternion


def train_neural_network():
    """Train a neural network on a classification dataset and save the trained model."""
    # Example training and test data
    train_dataset = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    train_inputs = train_dataset[0]
    train_targets = np.real(train_dataset[1]).reshape(-1, 1)  # Convert complex targets to real values

    test_dataset = datasets.make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    test_inputs = test_dataset[0]
    test_targets = np.real(test_dataset[1]).reshape(-1, 1)  # Convert complex targets to real values

    # Ensure binary targets
    train_targets = np.where(train_targets > 0, 1, 0)
    test_targets = np.where(test_targets > 0, 1, 0)

    # Create a neural network
    layers = [2, 4, 4, 4, 1]  # Updated number of layers
    nn = NeuralNetwork(*layers, activation='quaternion_sigmoid')

    # Train the neural network
    learning_rate = 0.1
    epochs = 10000
    nn.train(train_inputs, train_targets, test_inputs, test_targets, learning_rate=learning_rate, epochs=epochs)
    
    # Evaluate the model on the test set
    test_accuracy = nn.evaluate_model(test_inputs, test_targets)
    
    # Save the trained model
    nn.save_model("trained_model.npy")
    print("Model training complete. Trained model saved as 'trained_model.npy'.")

def load_pretrained_model():
    """Load a pre-trained model and make predictions using the loaded model."""
    try:
        # Load the pre-trained model
        loaded_model = NeuralNetwork.load_model("trained_model.npy")
        print("Pre-trained model loaded successfully.")

        # Predict using the loaded model
        dataset = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        inputs = dataset[0]
        predictions = loaded_model.predict(inputs)
        print("\nPredictions:")
        print(predictions)

    except FileNotFoundError:
        print("No pre-trained model found. Please train the model first.")


def perform_fourier_and_banach_operations():
    """Perform Fourier and Banach Space operations and print the results."""
    # Fourier and Banach Space operations
    # Create a signal and its corresponding Fourier class instance
    signal = np.sin(np.linspace(0, 10, num=100))
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

    # Create two BanachSpace instances
    banach_space_1 = BanachSpace([1, 2, 3])
    banach_space_2 = BanachSpace([4, 5, 6])

    # Perform operations on the instances
    norm_A = banach_space_1.norm()
    scaled_A = banach_space_1.scalar_multiply(2)
    sum_AB = banach_space_1.add(banach_space_2)
    dot_AB = banach_space_1.dot_product(banach_space_2)

    # Print the results
    print("BanachSpace A: ", banach_space_1)
    print("Norm of A: ", norm_A)
    print("Scaled A: ", scaled_A)
    print("Sum of A and B: ", sum_AB)
    print("Dot product of A and B: ", dot_AB)


def apply_operations_to_qnn():
    """Apply Fourier, Banach, and Hilbert operations to QNN and print the results."""
    # Apply Fourier, Banach, and Hilbert operations to QNN
    # Create a signal and its corresponding Fourier class instance
    signal = np.sin(np.linspace(0, 10, num=100))
    fourier = Fourier(signal)

    # Perform Fourier transform
    x = np.arange(len(signal))
    freq, spectrum = fourier.compute_fourier_transform(x, signal)

    # Analyze the spectrum
    analyzed_freq, analyzed_coefficients = fourier.analyze_fourier_coefficients(freq, spectrum, 5)

    # Create two BanachSpace instances
    banach_space_1 = BanachSpace([1, 2, 3])
    banach_space_2 = BanachSpace([4, 5, 6])

    # Perform operations on the instances
    norm_A = banach_space_1.norm()
    scaled_A = banach_space_1.scalar_multiply(2)
    sum_AB = banach_space_1.add(banach_space_2)
    dot_AB = banach_space_1.dot_product(banach_space_2)

    # Create a Hilbert space in 4 dimensions with quaternion values
    values = [Quaternion(1.0, 2.0, 3.0, 4.0), Quaternion(2.0, 3.0, 4.0, 5.0),
              Quaternion(3.0, 4.0, 5.0, 6.0), Quaternion(4.0, 5.0, 6.0, 7.0)]
    hilbert_space = HilbertSpace(values)

    # Perform operations on the Hilbert space
    norm = hilbert_space.norm()
    scaled = hilbert_space.scalar_multiply(2)
    summed = hilbert_space.add(HilbertSpace([Quaternion(1.0, 1.0, 1.0, 1.0),
                                             Quaternion(1.0, 1.0, 1.0, 1.0),
                                             Quaternion(1.0, 1.0, 1.0, 1.0),
                                             Quaternion(1.0, 1.0, 1.0, 1.0)]))
    dot_product = hilbert_space.dot_product(HilbertSpace([Quaternion(2.0, 3.0, 4.0, 5.0),
                                                          Quaternion(3.0, 4.0, 5.0, 6.0),
                                                          Quaternion(4.0, 5.0, 6.0, 7.0),
                                                          Quaternion(5.0, 6.0, 7.0, 8.0)]))

    # Print the results
    print("Fourier Transform of Signal: ", spectrum)
    print("Analyzed Frequencies: ", analyzed_freq)
    print("Analyzed Coefficients: ", analyzed_coefficients)
    print("Norm of BanachSpace A: ", norm_A)
    print("Scaled BanachSpace A: ", scaled_A)
    print("Sum of BanachSpaces A and B: ", sum_AB)
    print("Dot Product of BanachSpaces A and B: ", dot_AB)
    print("Norm of HilbertSpace: ", norm)
    print("Scaled HilbertSpace: ", scaled)
    print("Sum of HilbertSpace and Quaternion values: ", summed)
    print("Dot Product of HilbertSpace and Quaternion values: ", dot_product)


def main():
    """Main function to interact with the program and select the desired operations."""
    nn = None  # Variable to store the neural network object

    while True:
        print("\n1. Train the model")
        print("2. Load a pre-trained model")
        print("3. Fourier and Banach Operations")
        print("4. Apply Operations to QNN")
        print("5. Exit")

        choice = input("Enter your choice (1, 2, 3, 4, or 5): ")

        if choice == '1':
            train_neural_network()
            nn = NeuralNetwork.load_model("trained_model.npy")
            print("Pre-trained model loaded successfully.")

        elif choice == '2':
            load_pretrained_model()
            nn = NeuralNetwork.load_model("trained_model.npy")
            print("Pre-trained model loaded successfully.")

        elif choice == '3':
            perform_fourier_and_banach_operations()

        elif choice == '4':
            if nn is None:
                print("Please train or load a pre-trained model first.")
            else:
                apply_operations_to_qnn()

        elif choice == '5':
            print("Exiting the program...")
            break

        else:
            print("Invalid choice. Please enter a valid option.")



if __name__ == '__main__':
    main()
