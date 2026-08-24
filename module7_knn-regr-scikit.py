# Import NumPy for storing and processing the dataset
import numpy as np

# Import KNeighborsRegressor from Scikit-learn
from sklearn.neighbors import KNeighborsRegressor


# Ask the user for the number of data points
N = int(input("Enter a positive integer N: "))

# Ask the user for the number of nearest neighbors
k = int(input("Enter a positive integer k: "))

# Check if k is greater than N
if k > N:
    print("Error: k cannot be greater than N.")

else:
    # Create empty NumPy arrays for x-values and y-values
    x_values = np.empty(N)
    y_values = np.empty(N)

    # Read N (x, y) points from the user
    for i in range(N):
        x_values[i] = float(input(f"Enter x value for point {i + 1}: "))
        y_values[i] = float(input(f"Enter y value for point {i + 1}: "))

    # Ask the user for the X value to predict
    X = float(input("Enter X value for prediction: "))

    # Reshape x-values because Scikit-learn expects a 2D array
    X_train = x_values.reshape(-1, 1)

    # Create the k-NN regression model
    model = KNeighborsRegressor(n_neighbors=k)

    # Train the model using the input data
    model.fit(X_train, y_values)

    # Predict Y for the user's X value
    predicted_y = model.predict(np.array([[X]]))

    # Calculate the variance of the training labels
    label_variance = np.var(y_values)

    # Display the predicted Y value
    print("Predicted Y:", predicted_y[0])

    # Display the variance of the labels
    print("Variance of labels:", label_variance)
