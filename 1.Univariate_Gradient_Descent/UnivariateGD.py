from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

i = 1


def grad(x):
    return 2 * x + 5 * np.cos(x)


def cost(x):
    return x ** 2 + 5 * np.sin(x)


def myGD1(alpha, x0, gra=1e-3, loop=1000):
    global i
    it = None
    x = [x0]
    for it in range(loop):
        x_new = x[-1] - alpha * grad(x[-1])
        if abs(grad(x_new)) < gra:
            break
        x.append(x_new)
    plt.plot(x, [cost(x) for x in x])
    plt.scatter(x[-1], cost(x[-1]))
    plt.savefig(f'result{i}.png')
    plt.show()
    i += 1
    return x, it


if __name__ == '__main__':
    (x1, it1) = myGD1(0.1, -10)
    print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost(x1[-1]), it1))

    (x2, it2) = myGD1(0.1, 10)
    print('Solution x2 = %f, cost = %f, obtained after %d iterations' % (x2[-1], cost(x2[-1]), it2))

    (x3, it3) = myGD1(0.01, 10)
    print('Solution x4 = %f, cost = %f, obtained after %d iterations' % (x3[-1], cost(x3[-1]), it3))
