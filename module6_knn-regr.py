#import Numpy library
import numpy as np


class KNNRegression:
    def __init__(self, n, k):
        self.n = n
        self.k = k

        # Create NumPy arrays to store x and y values
        self.x_values = np.zeros(n)
        self.y_values = np.zeros(n)

    def insert_data(self, index, x, y):
        # Store x and y values in NumPy arrays
        self.x_values[index] = x
        self.y_values[index] = y

    def predict(self, x):
        # Check if k is greater than N
        if self.k > self.n:
            return None

        # Calculate distance between input X and every stored x value
        distances = np.abs(self.x_values - x)

        # Get indexes of the k smallest distances
        nearest_indices = np.argsort(distances)[:self.k]

        # Get y values of the k nearest points
        nearest_y_values = self.y_values[nearest_indices]

        # Calculate average y value
        prediction = np.mean(nearest_y_values)

        return prediction


# Ask user for N
N = int(input("Enter N: "))

# Ask user for k
k = int(input("Enter k: "))

# Check that N and k are positive
if N <= 0 or k <= 0:
    print("Error: N and k must be positive integers.")

elif k > N:
    print("Error: k cannot be greater than N.")

else:
    # Create the k-NN Regression object
    model = KNNRegression(N, k)

    # Read N (x, y) points
    for i in range(N):
        print(f"Point {i + 1}")

        x = float(input("Enter x: "))
        y = float(input("Enter y: "))

        model.insert_data(i, x, y)

    # Ask for the X value to predict
    X = float(input("Enter X for prediction: "))

    # Calculate predicted Y
    Y = model.predict(X)

    print("Predicted Y:", Y)
