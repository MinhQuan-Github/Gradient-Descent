import numpy as np
import matplotlib.pyplot as plt


def grad(x):
    return 2 * x + 10 * np.cos(x)


def cost(x):
    return x ** 2 + 10 * np.sin(x)


# check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 0.001


def GD_momentum(theta_init, alpha=0.1, beta=0.9):
    it = None
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(1000):
        v_new = beta * v_old + alpha * grad(theta[-1])
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        v_old = v_new
    plt.plot(theta, [cost(x) for x in theta])
    plt.savefig('momentumGD.png')
    plt.show()
    return theta, it


def myGD1(x0, alpha=0.1, gra=1e-3, loop=1000):
    it = None
    x = [x0]
    for it in range(loop):
        x_new = x[-1] - alpha * grad(x[-1])
        if abs(grad(x_new)) < gra:
            break
        x.append(x_new)
    plt.plot(x, [cost(x) for x in x])
    plt.savefig('normalGD.png')
    plt.show()
    return x, it


if __name__ == '__main__':
    (x3, it3) = myGD1(5, 0.01)
    print('GD_Solution x3 = %f, cost = %f, obtained after %d iterations' % (x3[-1], cost(x3[-1]), it3))
    (x1, it1) = GD_momentum(5, 0.1, beta=0.9)
    print('Momentum_Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost(x1[-1]), it1))
