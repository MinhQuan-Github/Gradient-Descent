from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style='ticks')


def next_batch(features, labels, batch_size):
    for data in range(0, np.shape(features)[0], batch_size):
        yield features[data: data + batch_size], labels[data: data + batch_size]


def stochastic_gradient_descent(X, y, alpha=0.01, epochs=100, batch_size=1):
    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    W = np.random.randn(n + 1, )

    cost_history_list = []

    for current_iteration in range(epochs):
        batch_epoch_loss_list = []

        for (X_batch, y_batch) in next_batch(X, y, batch_size):
            batch_m = np.shape(X_batch)[0]
            y_estimated = X_batch.dot(W)
            error = y_estimated - y_batch
            cost = (1 / 2 * m) * np.sum(error ** 2)
            batch_epoch_loss_list.append(cost)
            gradient = (1 / batch_m) * X_batch.T.dot(error)
            W = W - alpha * gradient

        print(f"cost:{np.average(batch_epoch_loss_list)} \t"
              f" iteration: {current_iteration}")

        cost_history_list.append(np.average(batch_epoch_loss_list))

    return W, cost_history_list


def run():
    rng = np.random.RandomState(20)
    X = 10 * rng.rand(1000, 5)
    y = 0.9 + np.dot(X, [2.2, 4., -4, 1, 2])

    weight, cost_history_list = stochastic_gradient_descent(
        X=X,
        y=y,
        alpha=0.001,
        epochs=10,
        batch_size=32
    )

    plt.plot(np.arange(len(cost_history_list)), cost_history_list)
    plt.xlabel("Number of iterations (Epochs)")
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Stochastic Gradient Descent")
    plt.savefig('StochasticGD.png')
    plt.show()


if __name__ == '__main__':
    run()
