# E B
# D C

from math import floor
from pandas import DataFrame
from tabulate import tabulate
from random import randint

import numpy as np
import matplotlib.pyplot as plt


class Matrix:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.__createMatrixA()
        self.__createMatrixF()

    def __createMatrixA(self):
        self.matrixA = []
        for i in range(self.n):
            self.matrixA.append([0] * self.n)
        for i in range(self.n):
            for j in range(self.n):
                self.matrixA[i][j] = randint(-10, 10)

    def createMatrixA(self, matrix):
        self.matrixA = []
        for i in range(self.n):
            self.matrixA.append([0] * self.n)
        for i in range(self.n):
            for j in range(self.n):
                self.matrixA[i][j] = matrix[i][j]
        self.__createMatrixF()

    def __createMatrixF(self):
        self.matrixF = []
        for i in range(self.n):
            self.matrixF.append([0] * self.n)
        for i in range(self.n):
            for j in range(self.n):
                self.matrixF[i][j] = self.matrixA[i][j]

    def showMatrixA(self):
        # for i in range(self.n):
        #     print(self.matrixA[i])
        df = DataFrame(self.matrixA)
        print(tabulate(df, headers='', tablefmt='rounded_grid', showindex=''))

    def showMatrixF(self):
        # for i in range(self.n):
        #     print(self.matrixF[i])
        df = DataFrame(self.matrixF)
        print(tabulate(df, headers='', tablefmt='rounded_grid', showindex=''))

    def __checkCondition1(self):
        evenNumbersCount = 0
        sumOddNumbers = 0
        for i in range(floor(self.n / 2), self.n):
            for j in range(floor(self.n / 2), self.n):
                if (j + 1) % 2 == 1 and self.matrixF[i][j] % 2 == 0:
                    evenNumbersCount += 1
                if (i + 1) % 2 == 1:
                    sumOddNumbers += self.matrixF[i][j]
        return (evenNumbersCount > sumOddNumbers)

    def __symmSwapCE(self):
        temp_e = []
        temp_c = []

        for i in range(floor(self.n / 2)):
            temp_e.append([0] * floor(self.n / 2))

        for i in range(floor(self.n / 2)):
            temp_c.append([0] * floor(self.n / 2))

        for i in range(floor(self.n / 2)):
            for j in range(floor(self.n / 2)):
                temp_e[i][j] = self.matrixF[i][j]

        for i in range(floor(self.n / 2), self.n):
            for j in range(floor(self.n / 2), self.n):
                temp_c[i - floor(self.n / 2)][j - floor(self.n / 2)] = self.matrixF[i][j]

        for i in range(floor(self.n / 2)):
            for j in range(floor(self.n / 2)):
                self.matrixF[j][i], self.matrixF[j][i] = temp_c[i][j], temp_c[j][i]
        for i in range(floor(self.n / 2), self.n):
            for j in range(floor(self.n / 2), self.n):
                self.matrixF[j][i], self.matrixF[i][j] = temp_e[i - floor(self.n / 2)][j - floor(self.n / 2)], \
                    temp_e[j - floor(self.n / 2)][i - floor(self.n / 2)]

    def __notSymmSwapBE(self):
        for i in range(floor(self.n / 2)):
            for j in range(floor(self.n / 2)):
                temp = self.matrixF[0][0]
                self.matrixF[i].append(self.matrixF[i].pop(0))

    def changeMatrixF(self):
        if self.__checkCondition1() == True:
            self.__symmSwapCE()
        if self.__checkCondition1() == False:
            self.__notSymmSwapBE()

    def __checkCondition2(self):
        matrADet = np.linalg.det(self.matrixA)
        print("Определитель матрицы A равен ", matrADet)
        sumDiagF = np.trace(self.matrixF)
        print("Сумма диагональных элементов матрицы F равна  ", sumDiagF)
        return matrADet > sumDiagF

    def calc_expressions(self):
        matrAnp = np.array(self.matrixA)
        matrFnp = np.array(self.matrixF)
        matrGnp = np.array(self.matrixA)
        if self.__checkCondition2():
            result = matrAnp.dot(np.transpose(matrAnp)) - self.k * np.linalg.matrix_power(matrFnp, -1)
            print('Результат выражения 1:')
            print(result)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    if j > i:
                        matrGnp[i][j] = 0
            result = (np.transpose(matrAnp) + matrGnp - np.transpose(matrFnp)) * self.k
            print('Результат выражения 2:')
            print(result)

    def graph1(self):
        zeroCounter = 0
        positiveCounter = 0
        negativeCounter = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.matrixF[i][j] == 0:
                    zeroCounter += 1
                if self.matrixF[i][j] > 0:
                    positiveCounter += 1
                if self.matrixF[i][j] < 0:
                    negativeCounter += 1
        x = [zeroCounter, positiveCounter, negativeCounter]
        plt.title('Распределение значений матрицы F')
        plt.pie(x, labels=['Ноль', 'Положительные числа', 'Отрицательные числа'])
        plt.show()

    def graph2(self):
        N = floor(self.n / 2)
        matrF = np.array(self.matrixF)
        eSum = np.sum(matrF[:N, :N])
        bSum = np.sum(matrF[:N, N:])
        dSum = np.sum(matrF[N:, :N])
        cSum = np.sum(matrF[N:, N:])
        plt.title('Сумма значений элементов в подматрицах')
        plt.bar([1, 2, 3, 4], [eSum, bSum, dSum, cSum])
        plt.show()

    def graph3(self):
        matrF = np.array(self.matrixF)
        plt.imshow(matrF)
        plt.show()


try:
    k = int(input('Введите K:'))
    n = int(input('Введите N:'))
    matrix = Matrix(k, n)
    print('Исходная матрица А:')
    matrix.showMatrixA()
    matrix.changeMatrixF()
    print('Измененная матрица F:')
    matrix.showMatrixF()
    matrix.calc_expressions()
    matrix.graph1()
    matrix.graph2()
    matrix.graph3()
except:
    pass
