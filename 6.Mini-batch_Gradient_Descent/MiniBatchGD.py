import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def miniBatchGD(X, y, batch_size=20, learn_rate=0.005, num_iter=25):
    n_points = X.shape[0]
    W = np.zeros(X.shape[1])
    b = 0

    regression = [np.hstack((W, b))]
    for _ in range(num_iter):
        batch = np.random.choice(range(n_points), batch_size)
        X_batch = X[batch, :]
        y_batch = y[batch]
        W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
        regression.append(np.hstack((W, b)))

    return regression


def MSEStep(X, y, W, b, learn_rate=0.005):
    y_pred = np.matmul(X, W) + b

    error = y - y_pred

    W_new = W + learn_rate * np.matmul(error, X)
    b_new = b + learn_rate * error.sum()

    return W_new, b_new


if __name__ == "__main__":
    data = np.loadtxt('data.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    regression_coef = miniBatchGD(X, y)

    plt.figure()
    X_min = X.min()
    X_max = X.max()
    counter = len(regression_coef)
    for W, b in regression_coef:
        counter -= 1
        color = [1 - 0.92 ** counter for _ in range(3)]
        plt.plot([X_min, X_max], [X_min * W + b, X_max * W + b], color=color)
    plt.scatter(X, y, zorder=3)
    plt.savefig('MiniBatchGD.png')
    plt.show()
