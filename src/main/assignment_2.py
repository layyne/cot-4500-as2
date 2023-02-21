import numpy as np
from math import prod
from interpolation import neville, newton_forward, hermite, cubic_spline


# Question 1
def q1():
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    p = 3.7

    return neville(x, y, p)


# Question 2
def q2():
    xs = [7.2, 7.4, 7.5, 7.6]
    ys = [23.5492, 25.3913, 26.8224, 27.4589]

    coeff, _ = newton_forward(xs, ys)
    return list(coeff)


# Question 3
def q3():
    xs = [7.2, 7.4, 7.5, 7.6]
    ys = [23.5492, 25.3913, 26.8224, 27.4589]
    x = 7.3

    _, p = newton_forward(xs, ys)
    return p(x)


# Question 4
def q4():
    xs = [3.6, 3.8, 3.9]
    ys = [1.675, 1.436, 1.318]
    dys = [-1.195, -1.188, -1.182]

    return hermite(xs, ys, dys)


# Question 5
def q5():
    xs = [2, 5, 8, 10]
    fxs = [3, 5, 7, 9]

    return cubic_spline(xs, fxs)


if __name__ == '__main__':
    print(q1(), end='\n\n')

    print(q2(), end='\n\n')

    print(q3(), end='\n\n')

    # Figured this out on my own before I noticed you already gave us this line :(
    np.set_printoptions(linewidth=100, precision=7, suppress=True)

    print(q4(), end='\n\n')

    print(*q5(), sep='\n\n', end='\n\n')
