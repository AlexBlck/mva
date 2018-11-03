import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
import datetime

class Gaussian:
    def __init__(self, mean, std, points=1000, noise=0.001):
        self.noise = noise
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

    def __truediv__(self, other):
        xmin = min(self.xmin, other.xmin)
        xmax = max(self.xmax, other.xmax)
        delx = xmax - xmin
        step = delx / (self.points + other.points)
        x = [x for x in np.arange(xmin, xmax, step)]
        y = [self.nd(y, self.mean, self.std) / np.sqrt(self.nd(y, other.mean, other.std)) for y in x]
        return x, y

    def nd(self, x, mean, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.e ** (((x - mean) ** 2) / (-2 * std ** 2)) + uniform(0, self.noise)

    def plot(self):
        plt.plot(self.x, self.y)


class Poisson:
    def __init__(self, events, noise=0.05):
        self.xmax = events * 3
        self.events = events
        self.noise = noise
        self.x = [x for x in np.arange(0, self.xmax)]
        self.y = [self.pd(x, events) for x in self.x]

    def __add__(self, other):
        xmax = max(self.xmax, other.xmax)
        x = [x for x in np.arange(0, xmax)]
        y = [self.pd(y, self.events) + self.pd(y, other.events) for y in x]
        return x, y

    def __truediv__(self, other):
        xmax = min(self.xmax, other.xmax)
        x = [x for x in np.arange(0, xmax)]
        y = [self.pd(y, self.events) / np.sqrt(self.pd(y, other.events)) for y in x]
        return x, y

    def pd(self, x, events):
        return np.e ** (-events) * events ** x / np.math.factorial(x) + uniform(0, self.noise)

    def plot(self):
        plt.plot(self.x, self.y, '.')


def bisection(x, y, thresh=1):
    x = np.array(x)
    y = np.array(y)
    a = x[0]
    b = x[-1]
    while (b - a) > thresh:

        m = (a + b) / 2
        l = (a + m) / 2
        r = (b + m) / 2

        index_f_a, a = min(enumerate(x), key=lambda x: abs(x[1]-a))
        index_f_b, b = min(enumerate(x), key=lambda x: abs(x[1]-b))
        index_f_m, m = min(enumerate(x), key=lambda x: abs(x[1]-m))
        index_f_l, l = min(enumerate(x), key=lambda x: abs(x[1]-l))
        index_f_r, r = min(enumerate(x), key=lambda x: abs(x[1]-r))

        f_a = y[index_f_a]
        f_b = y[index_f_b]
        f_m = y[index_f_m]
        f_l = y[index_f_l]
        f_r = y[index_f_r]

        maximum = np.max([f_a, f_b, f_m, f_l, f_r])

        if maximum == f_a or maximum == f_l:
            b = m
        elif maximum == f_b or maximum == f_r:
            a = m
        else:
            a = l
            b = r
    return m, f_m


for r, b in [[r,b] for r in range(1, 20) for b in range(1, 20)]:
    maxes = []
    for _ in range(1000):
        real = Poisson(r)
        bg = Poisson(b)
        real.plot()
        bg.plot()

        #sumx, sumy = real + bg
        sigx, sigy = real / bg
        maxx, maxy = bisection(sigx, sigy)
        maxes.append(maxx)

    plt.xlim(0, sigx[-1])

    y, x, _ = plt.hist(maxes, [x for x in range(sigx[-1])])
    plt.title("Signal: " + str(r) + ", Background:" + str(b) + ", tallest: " + str(np.where(y == y.max())[0]))

    now = datetime.datetime.now()
    date = "_".join([str(now.day), str(now.month), str(now.hour), str(now.minute), str(now.second)]) + '.png'
    plt.savefig('figures/' + date)
    plt.cla()
    # plt.show()


def plot_poisson():
    plt.plot(sigx, sigy, '--', color='gray')
    plt.plot(sumx, sumy, '--', color='lightgray')
    plt.plot(maxx, maxy, 'o', color='darkgray')
    plt.xlim(0, sigx[-1])
    plt.show()