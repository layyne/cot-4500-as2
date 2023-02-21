import numpy as np
from numpy.polynomial import Polynomial
from math import prod


def neville(xs, ys, p):
    n = len(xs)

    diffs = np.zeros((n, n))
    diffs[:, 0] = ys

    for i in range(1, n):
        for j in range(1, i + 1):
            t1 = (p - xs[i-j]) * diffs[i][j-1]
            t2 = (p - xs[i]) * diffs[i-1][j-1]
            diffs[i][j] = (t1 - t2) / (xs[i] - xs[i-j])

    return diffs[-1][-1]


def q1():
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    p = 3.7

    return neville(x, y, p)


def newton_forward(xs, ys):
    n = len(xs)

    diffs = np.zeros((n, n))
    diffs[:, 0] = ys

    for i in range(1, n):
        for j in range(1, i + 1):
            diffs[i][j] = (diffs[i][j-1] - diffs[i-1][j-1]) / (xs[i] - xs[i-j])

    coeff = diffs.diagonal()

    def p(x: float) -> float:
        return sum(coeff[i] * prod(x - xs[j] for j in range(i)) for i in range(n))

    return coeff[1:], p


def q2():
    xs = [7.2, 7.4, 7.5, 7.6]
    ys = [23.5492, 25.3913, 26.8224, 27.4589]

    coeff, _ = newton_forward(xs, ys)
    return coeff


def q3():
    xs = [7.2, 7.4, 7.5, 7.6]
    ys = [23.5492, 25.3913, 26.8224, 27.4589]
    x = 7.3

    _, p = newton_forward(xs, ys)
    return p(x)


def hermite(xs, ys, dys):
    zs = np.repeat(xs, 2)
    n = len(zs)

    diffs = np.zeros((n, n))
    diffs[:, 0] = np.repeat(ys, 2)

    # Set up column 2 (with derivative samples)
    for i in range(1, n):
        if i % 2 == 1:
            diffs[i][1] = dys[i//2]
        else:
            diffs[i][1] = (diffs[i][0] - diffs[i-1][0]) / (zs[i] - zs[i-1])

    # Perform divided difference iteration
    for i in range(2, n):
        for j in range(2, i + 1):
            diffs[i][j] = (diffs[i][j-1] - diffs[i-1][j-1]) / (zs[i] - zs[i-j])

    # Prepend z values I guess?
    result = np.append(np.array([zs]).T, diffs, axis=1)
    return result


def q4():
    xs = [3.6, 3.8, 3.9]
    ys = [1.675, 1.436, 1.318]
    dys = [-1.195, -1.188, -1.182]

    return hermite(xs, ys, dys)


if __name__ == '__main__':
    print(q1(), end='\n\n')

    print(list(q2()), end='\n\n')

    print(q3(), end='\n\n')

    np.set_printoptions(
        precision=7,
        suppress=True,
    )
    print(q4()[:, :-1], end='\n\n')
