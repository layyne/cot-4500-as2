import numpy as np
from math import prod


# Performs Neville's interpolation to approximate
# f(p) given a set of ordered pairs
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


# Performs Newton's forward difference method to produce a set of coefficients
# and corresponding interpolating polynomial given a set of ordered pairs
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


# Performs Hermite interpolation to produce an
# approximation matrix given a set of ordered pairs
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
    return result[:, :-1]


# Performs cubic spline interpolation to produce the interpolating
# matrix equation and solution for a given set of ordered pairs
def cubic_spline(xs, fxs):
    xs = np.float64(xs)
    fxs = np.float64(fxs)

    # So, so awkward
    n = len(xs) - 1

    hs = [xs[i+1] - xs[i] for i in range(n)]

    alphs = [
        (3 / hs[i]) * (fxs[i+1] - fxs[i]) - (3 / hs[i-1]) * (fxs[i] - fxs[i-1])
        for i in range(1, n)
    ]

    ls = [1]
    mus = [0]
    zs = [0]

    for i in range(1, n):
        ls.append(2 * (xs[i+1] - xs[i-1]) - hs[i-1] * mus[i-1])
        mus.append(hs[i] / ls[i])
        zs.append((alphs[i-1] - hs[i-1] * zs[i-1]) / ls[i])

    ls.append(1)
    zs.append(0)

    bs = [0] * n
    cs = [0] * (n + 1)
    ds = [0] * n

    for i in reversed(range(n)):
        cs[i] = zs[i] - mus[i] * cs[i+1]
        bs[i] = (fxs[i+1] - fxs[i]) / hs[i] - hs[i] * (cs[i+1] + 2 * cs[i]) / 3
        ds[i] = (cs[i+1] - cs[i]) / (3 * hs[i])

    # Build matrix A
    A = np.float64([[0] * (n + 1) for _ in range(n + 1)])
    for i in range(n - 1):
        A[i+1][i:i+3] = hs[i], 2 * (hs[i] + hs[i+1]), hs[i+1]
    A[0][0], A[-1][-1] = 1, 1

    # Vectors b and x (solution)
    b = np.float64([0] + alphs + [0])
    x = np.float64(cs)

    return A, b, x
