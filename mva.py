import numpy as np
# from numpy.random import normal
import matplotlib.pyplot as plt


class Gaussian:
    def __init__(self, mean, std, points=1000):
        self.mean = mean
        self.std = std
        self.xmin = mean - 4 * std
        self.xmax = mean + 4 * std
        self.delx = self.xmax - self.xmin
        self.points = points
        self.step = self.delx / points
        self.x = [x for x in np.arange(self.xmin, self.xmax, self.step)]
        self.y = [self.nd(x, mean, std) for x in self.x]

    def __add__(self, other):
        xmin = min(self.xmin, other.xmin)
        xmax = max(self.xmax, other.xmax)
        delx = xmax - xmin
        step = delx / (self.points + other.points)
        x = [x for x in np.arange(xmin, xmax, step)]
        y = [self.nd(y, self.mean, self.std) + self.nd(y, other.mean, other.std) for y in x]
        return x, y

    def nd(self, x, mean, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.e ** (((x - mean) ** 2) / (-2 * std ** 2))

    def plot(self):
        plt.plot(self.x, self.y)


def create_and_plot():
    background = Gaussian(10, 2, 1000)
    signal = Gaussian(15, 5, 1000)

    background.plot()
    signal.plot()
    plt.show()
    x, y = background + signal
    plt.plot(x, y)
    plt.show()


def create():
    background = Gaussian(10, 2, 1000)
    signal = Gaussian(15, 5, 1000)
    x, y = background + signal
    x = np.array(x)
    y = np.array(y)
    return x, y


def bisection(x, y, thresh=0.001):
    a = x[0]
    b = x[-1]
    while (b - a) > thresh:

        f_a = y[len(*np.where(x < a))]
        f_b = y[len(*np.where(x < b))]

        m = (a + b) / 2
        f_m = y[len(*np.where(x < m))]

        l = (a + m) / 2
        f_l = y[len(*np.where(x < l))]

        r = (b + m) / 2
        f_r = y[len(*np.where(x < r))]

        maximum = max(f_a, f_b, f_m, f_l, f_r)

        if maximum == f_a or maximum == f_l:
            b = m
        elif maximum == f_b or maximum == f_r:
            a = m
        else:
            a = l
            b = r
    return m, f_m


x, y = create()

m, f_m = bisection(x, y)

plt.plot(x, y)
plt.plot(m, f_m, 'o')
plt.show()