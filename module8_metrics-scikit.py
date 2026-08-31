import numpy as np
from sklearn.metrics import precision_score, recall_score

# Ask the user for the number of data points
N = int(input("Enter N (positive integer): "))

# Check that N is a positive integer
if N <= 0:
    print("Error: N must be a positive integer.")

else:
    # Create NumPy arrays to store the ground truth and predicted class labels
    y_true = np.zeros(N, dtype=int)
    y_pred = np.zeros(N, dtype=int)

    # Ask the user to enter N pairs of class labels
    for i in range(N):
        print(f"Point {i + 1}")

        # Read the ground truth class label X
        x = int(input("Enter X (ground truth: 0 or 1): "))

        # Make sure X is either 0 or 1
        while x not in [0, 1]:
            print("Error: X must be either 0 or 1.")
            x = int(input("Enter X (ground truth: 0 or 1): "))

        # Read the predicted class label Y
        y = int(input("Enter Y (predicted class: 0 or 1): "))

        # Make sure Y is either 0 or 1
        while y not in [0, 1]:
            print("Error: Y must be either 0 or 1.")
            y = int(input("Enter Y (predicted class: 0 or 1): "))

        # Store the values in the NumPy arrays
        y_true[i] = x
        y_pred[i] = y

    # Calculate Precision using Scikit-learn
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Calculate Recall using Scikit-learn
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Display the results
    print("\nResults:")
    print("Precision:", precision)
    print("Recall:", recall)
