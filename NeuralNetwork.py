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


# кол-во входных, скрытых и выходных слоев
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения
learning_rate = 0.1

# экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузка в список тестовый набор данных CSV-файла набора MNIST
training_data_file = open("mnist_dataset/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренировка нейронной сети

# переменная epochs указывает, сколько раз тренировочный набор данных используется для тренировки сети
epochs = 5

for _ in range(epochs):
    # перебираем все записи в тренировочном наборе
    for record in training_data_list:
        # получаем список значений, используя символы запятой (',') в качестве разделтеля
        all_values = record.split(',')
        # масштабируем и смещаем входные значения
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        # создаем целевые выходные значения (все равны 0.01, желаемое - 0.99)
        targets = np.zeros(output_nodes) + 0.01

        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        # plt.imshow(image_array, cmap='Greys', interpolation='None')
        # plt.show()


# обучение по своим картинкам
# img_array = scipy.misc.imread(image_file, flatten=True)
# img_data = 255 - img_array.reshape(784)
# img_data = (img_data / 255 * 0.99) + 0.01


# создание перевернутых на некоторый угол вариантов изображений
# повернуть на 10 градусов против часовой стрелки
# inputs_plus10_img = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28, 28), 10, cval=0.01, reshape=False)
# повернуть на 10 градусов по часовой стрелки
# inputs_plus10_img = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28, 28), -10, cval=0.01, reshape=False)
# cval - смещение на 0.01, чтобы на вход нейронки не подавались нули


# загрузка в список тестового набора данных CSV-файла MNIST
test_data_file = open("mnist_dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# тестирование нейронки

# журнал оценок работы сети, первоначально пустой
scorecard = []

# перебрать все записи в тестовом наборе данных
for record in test_data_list:
    # получаем список значений из записи, используя символы запятой (',') в качестве разделителя
    all_values = record.split(',')
    # print(all_values)
    # правильный ответ - первое значение
    correct_label = int(all_values[0])
    print(correct_label, "истинный маркер")
    # масштабируем и смещаем входные значения
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    # print(outputs)
    # индекс наибольшего значения - это маркерное значение
    label = np.argmax(outputs)
    print(label, "ответ сети")
    # присоединить оценку ответа сети к концу списка
    if label == correct_label:
        # в случае правильного ответа сети присоединить к списку 1
        scorecard.append(1)
    else:
        # в случае неправильного ответа - 0
        scorecard.append(0)

# рассчитать показатель эффективности в виде доли правильных ответов
scorecard_array = np.asarray(scorecard)  # преобразование в numpy массив
print("Эффективность = ", scorecard_array.sum() / scorecard_array.size)








