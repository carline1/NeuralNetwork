import numpy as np
# import math
import scipy.special
from matplotlib import pyplot as plt
import scipy.misc


class neuralNetwork:

    # инициализация нейронки
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задаем кол-во нейронов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # задаем коэффициент обучения
        self.lr = learningrate

        # простые весовые коэффициенты (генерируем массив с рандомными весами от -0.5 до 0.5)
        # матрица имеет вид w11 w21
        #                   w12 w22
        # # веса между входным и скрытым слоями
        # self.wih = np.random.rand(self.hnodes, self.inodes)
        # # веса между скрытым и выходным слоями
        # self.who = np.random.rand(self.onodes, self.hnodes)

        # ???????????????????????
        # улучшенная генерация весов (нормально распределение с центром в нуле и со стандартным отклонением, равным
        # 1/sqrt(кол-во входящих связей на узел))
        # веса между входным и скрытым слоями
        self.wih = np.random.normal(0.0, self.hnodes ** -0.5, (self.hnodes, self.inodes))
        # веса между скрытым и выходным слоями
        self.who = np.random.normal(0.0, self.onodes ** -0.5, (self.onodes, self.hnodes))

        # функция активации
        # (потом реализую без библиотеки)
        # аналог через библиотеку scipy.special
        self.activation_function = lambda x: scipy.special.expit(x)

    # обучение нейронки
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двумерный массив (вертикальный 3x1)
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)  # перемножение матриц
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)  # перемножение матриц
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                    np.transpose(hidden_outputs))

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                    np.transpose(inputs))

    # опрос нейронки
    def query(self, inputs_list):
        # преобразовать список входных значений в двумерный массив (вертикальный 3x1)
        inputs = np.array(inputs_list, ndmin=2).T
        # print(inputs)

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)  # перемножение матриц
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)  # перемножение матриц
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs