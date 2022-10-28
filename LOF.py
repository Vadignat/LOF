import math
import numpy as np

class LOF:
    def __init__(self, X, k_neighbours=2):
        self.__k = k_neighbours
        self.__X = X

    def fit(self):
        return np.array([self.__LOF(x) for x in self.__X])

    def __dist(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum += (x[i] - y[i])**2
        sum = math.sqrt(sum)
        return sum

    def __k_dist(self, a):
        pd = [self.__dist(a, x) for x in self.__X if 0 < self.__dist(a, x)]
        pd = sorted(pd)
        return pd[self.__k - 1]

    def __N_k(self, a):
        return[x for x in self.__X if 0 < self.__dist(a, x) <= self.__k_dist(a)]

    def __RD(self, a, b):
        return max(self.__k_dist(b), self.__dist(a, b))

    def __LRD(self, a):
        sum = 0
        for x_j in self.__N_k(a):
            sum += self.__RD(a, x_j)
        sum = len(self.__N_k(a)) / sum
        return sum

    def __LOF(self, a):
        sum = 0
        for x_j in self.__N_k(a):
            sum += self.__LRD(x_j)
        sum = sum/(len(self.__N_k(a)) * self.__LRD(a))
        return sum

