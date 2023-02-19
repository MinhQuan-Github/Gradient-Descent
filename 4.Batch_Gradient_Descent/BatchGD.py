# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Define the loss function
def loss_function(x):
    return x ** 2 + 10 * np.sin(x)


# Define the gradient of the loss function
def gradient_function(x):
    return 2 * x + 10 * np.cos(x)


# Initialize the learning rate and the number of iterations
learning_rate = 0.1
num_iterations = 100

# Initialize the starting point
x = 5
losses = []


def BatchGD():
    global x
    # Implement batch gradient descent
    for i in range(num_iterations):
        # Calculate the gradient of the loss function
        gradient = gradient_function(x)

        # Update the parameters using the gradient and the learning rate
        x = x - learning_rate * gradient

        # Print the current value of the loss function
        loss = loss_function(x)
        losses.append([x, loss])
        print("Iteration ", i + 1, ": Loss = ", loss)


if __name__ == '__main__':
    BatchGD()
    losses = np.array(losses)
    plt.plot(losses[:, 0], losses[:, 1])
    plt.savefig('batchGD.png')
    plt.show()
