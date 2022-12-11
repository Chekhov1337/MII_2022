import math
import numpy
import matplotlib.pyplot as plt
import csv
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# region Метод КНН для одной точки (с Парзеновским окном и k)
def knn(learn_samples, samples_to_check, window_size, k, samples_type='new'):
    checked_points = []
    if window_size <= 0:
        window_size = 1
    if k <= 0:
        k = 1
    if samples_type == 'new':
        print('Выборка с новыми продуктами:')
        for m in range(0, len(samples_to_check)):
            # region Определяем дистанцию от текущей точки до каждой точки в обучающей выборке
            distance = []
            classes_weight = [-1, -1, -1, -1]
            for j in range(0, len(learn_samples)):
                dist = math.sqrt((samples_to_check[m][0] - learn_samples[j][0]) ** 2 + (
                        samples_to_check[m][1] - learn_samples[j][1]) ** 2)
                distance.append([dist, learn_samples[j][2]])
            distance.sort(key=lambda x: x[0])
            while len(distance) > k:
                distance.pop(len(distance) - 1)
            # endregion
            for h in range(0, len(distance)):
                if distance[h][0] <= window_size:
                    if distance[h][0] == 0:
                        distance[h][0] = 0.00001
                        classes_weight[distance[h][1]] += 1 / distance[h][0]
                    else:
                        classes_weight[distance[h][1]] += 1 / distance[h][0]
            if classes_weight[0] == - 1 and classes_weight[1] == - 1 and classes_weight[2] == - 1:
                checked_points.append(
                    [samples_to_check[m][0], samples_to_check[m][1], -1])
            else:
                checked_points.append(
                    [samples_to_check[m][0], samples_to_check[m][1],
                     classes_weight.index(numpy.max(classes_weight))])
            temp_class = dict_class_names.get(checked_points[m][2])
            if classes_weight[0] != - 1 or classes_weight[1] != - 1 or classes_weight[2] != - 1:
                print(
                    '______________\nПродукт №{}\nСладость продукта: {}\nХруст продукта: {}'.format(
                        m + 1, checked_points[m][0], checked_points[m][1]))
                print('Класс продукта:', temp_class)

                sladost_x.append(checked_points[m][0])
                xpyct_y.append(checked_points[m][1])
                colors.append(dict_colors.get(checked_points[m][2]))
            else:
                print(
                    '______________\nПродукт №{}\nСладость продукта: {}\nХруст продукта: {}'.format(
                        m + 1, checked_points[m][0], checked_points[m][1]))
                print('Класс продукта:', temp_class)
                sladost_x.append(checked_points[m][0])
                xpyct_y.append(checked_points[m][1])
                colors.append('#ffffff')
        plt.subplot(1, 2, 2)
        plt.scatter(sladost_x, xpyct_y, c=colors)
        plt.xlabel('Сладость')
        plt.ylabel('Хруст')
        plt.title('Новая точка')
        plt.axis([0, 10, 0, 10])
        # c = plt.Circle((new_point[0], new_point[1]), radius=5, color='blue', alpha=0.1)
        # plt.gca().add_artist(c)
        plt.show()
    elif samples_type == 'test':
        mistakes_count = 0
        print('\n\nТестовая выборка:')
        for m in range(0, len(samples_to_check)):
            distance = []
            classes_weight = [-1, -1, -1, -1]
            for j in range(0, len(learn_samples)):
                dist = math.sqrt((samples_to_check[m][0] - learn_samples[j][0]) ** 2 + (
                        samples_to_check[m][1] - learn_samples[j][1]) ** 2)
                distance.append([dist, learn_samples[j][2]])
            distance.sort(key=lambda x: x[0])
            while len(distance) > k:
                distance.pop(len(distance) - 1)
            for h in range(0, len(distance)):
                if distance[h][0] <= window_size:
                    if distance[h][0] == 0:
                        distance[h][0] = 0.00001
                        classes_weight[distance[h][1]] += 1 / distance[h][0]
                    else:
                        classes_weight[distance[h][1]] += 1 / distance[h][0]
            if classes_weight[0] == - 1 and classes_weight[1] == - 1 and classes_weight[2] == - 1:
                checked_points.append(
                    [samples_to_check[m][0], samples_to_check[m][1], -1,
                     samples_to_check[m][2]])
            else:
                checked_points.append(
                    [samples_to_check[m][0], samples_to_check[m][1], classes_weight.index(numpy.max(classes_weight)),
                     samples_to_check[m][2]])
            if checked_points[m][2] != checked_points[m][3]:
                mistakes_count += 1

            temp_class = dict_class_names.get(checked_points[m][3])
            received_class = dict_class_names.get(checked_points[m][2])

            print(
                '______________\nПродукт №{}\nСладость продукта: {}\nХруст продукта: {}'.format(
                    m + 1, checked_points[m][0], checked_points[m][1]))
            print('Полученный класс продукта:', received_class)
            print('Заданный класс продукта:', temp_class)
        # print(checked_points)
        print('\n\nКоличество продуктов:', len(samples_to_check))
        print('Количество ошибок:', mistakes_count)
        print('Доля ошибок: {:.2f}%'.format(mistakes_count * 100 / len(samples_to_check)))


# endregion
# region Метод КНН Склерн
def knn_sklearn(k, dataset, test_samples_count):
    x = dataset.iloc[:, 0:2].values
    y = dataset.iloc[:, 2].values

    # обучающая и тестовая выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_samples_count / len(y), shuffle=False,
                                                        stratify=None)
    # метод knn из библиотеки sklearn с k соседей
    classifier = KNeighborsClassifier(n_neighbors=k)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # обучение
    classifier.fit(x_train, y_train)
    # проверка на тестовой выборке
    y_prediction = classifier.predict(x_test)
    # матрицы для представления результатов
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))


# endregion
# region Заполнение цсв файла (3 класса)
with open('Data.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\r')
    # 0 - Фрукт
    # 1 - Овощ
    # 2 - Протеин
    writer.writerow(['Продукт', 'Сладость', 'Хруст', 'Класс'])
    writer.writerow(['Яблоко', 9, 7, 0])
    writer.writerow(['Салат', 2, 5, 1])
    writer.writerow(['Бекон', 1, 2, 2])
    writer.writerow(['Банан', 9, 1, 0])
    writer.writerow(['Орехи', 1, 5, 2])
    writer.writerow(['Рыба', 1, 1, 2])
    writer.writerow(['Сыр', 1, 1, 2])
    writer.writerow(['Виноград', 8, 1, 0])
    writer.writerow(['Морковь', 2, 8, 1])
    writer.writerow(['Апельсин', 6, 1, 0])
# endregion
# region Чтение из цсв файла в датафрейм
df = pandas.read_csv('Data.csv', header=0, index_col=0, encoding='cp1251')
# endregion
# region Вывод графика с начальными данными
sladost_x = []
xpyct_y = []
colors = []

for row in df['Сладость']:
    sladost_x.append(row)
for row in df['Хруст']:
    xpyct_y.append(row)

dict_colors = {
    0: '#3cff89',
    1: '#9f3100',
    2: '#404040'
}
dict_class_names = {
    -1: 'Не определен',
    0: 'Фрукт',
    1: 'Овощ',
    2: 'Протеин'
}

for row in df['Класс']:
    colors.append(dict_colors.get(row))
plt.subplot(1, 2, 1)
plt.axis([0, 10, 0, 10])
plt.scatter(sladost_x, xpyct_y, c=colors)
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title('Изначальная выборка')
# plt.show()
# endregion
# region Заполнение массива с элементами выборки
learn_samples = []
for row in df['Сладость']:
    learn_samples.append([row, 0, 0])
i = 0
for row in df['Хруст']:
    learn_samples[i][1] = row
    i += 1
i = 0
for row in df['Класс']:
    learn_samples[i][2] = row
    i += 1

# endregion
# Вызовы методов для 3 классов
# Вызов метода для новых продуктов
new_samples = [[8, 2, 0], [2, 2, 2], [3, 8, 1]]
knn(learn_samples, new_samples, 5, 5)
# Вызов метода для тестовой выборки
test_samples = [[8, 2, 0], [2, 2, 2], [3, 8, 1]]
knn(learn_samples, test_samples, 5, 5, 'test')

# region Заполнение цсв файла для СК ЛЕРН (3 класса)
with open('Data.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\r')
    # 0 - Фрукт
    # 1 - Овощ
    # 2 - Протеин
    writer.writerow(['Продукт', 'Сладость', 'Хруст', 'Класс'])
    writer.writerow(['Яблоко', 9, 7, 0])
    writer.writerow(['Салат', 2, 5, 1])
    writer.writerow(['Бекон', 1, 2, 2])
    writer.writerow(['Банан', 9, 1, 0])
    writer.writerow(['Орехи', 1, 5, 2])
    writer.writerow(['Рыба', 1, 1, 2])
    writer.writerow(['Сыр', 1, 1, 2])
    writer.writerow(['Виноград', 8, 1, 0])
    writer.writerow(['Морковь', 2, 8, 1])
    writer.writerow(['Апельсин', 6, 1, 0])
    # test
    writer.writerow(['Мандарин', 8, 2, 0])
    writer.writerow(['Шашлык', 2, 2, 2])
    writer.writerow(['Капуста', 3, 8, 1])
# endregion
# region Чтение из цсв файла в датафрейм
df = pandas.read_csv('Data.csv', header=0, index_col=0, encoding='cp1251')
# endregion
# Вызов метода кнн склерн
knn_sklearn(5, df, 3)

# region Заполнение цсв файла (4 класса)
with open('Data.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\r')
    # 0 - Фрукт
    # 1 - Овощ
    # 2 - Протеин
    writer.writerow(['Продукт', 'Сладость', 'Хруст', 'Класс'])
    writer.writerow(['Яблоко', 9, 7, 0])
    writer.writerow(['Салат', 2, 5, 1])
    writer.writerow(['Бекон', 1, 2, 2])
    writer.writerow(['Банан', 9, 1, 0])
    writer.writerow(['Орехи', 1, 5, 2])
    writer.writerow(['Рыба', 1, 1, 2])
    writer.writerow(['Сыр', 1, 1, 2])
    writer.writerow(['Виноград', 8, 1, 0])
    writer.writerow(['Морковь', 2, 8, 1])
    writer.writerow(['Апельсин', 6, 1, 0])
    writer.writerow(['Леденец', 7, 9, 3])
    writer.writerow(['Шоколад', 9, 6, 3])
    writer.writerow(['Безе', 10, 10, 3])
    writer.writerow(['Вафля', 7, 8.5, 3])
# endregion
# region Чтение из цсв файла в датафрейм
df = pandas.read_csv('Data.csv', header=0, index_col=0, encoding='cp1251')
# endregion
# region Вывод графика с начальными данными
sladost_x = []
xpyct_y = []
colors = []

for row in df['Сладость']:
    sladost_x.append(row)
for row in df['Хруст']:
    xpyct_y.append(row)

dict_colors = {
    0: '#3cff89',
    1: '#9f3100',
    2: '#404040',
    3: '#9600ff'

}
dict_class_names = {
    -1: 'Не определен',
    0: 'Фрукт',
    1: 'Овощ',
    2: 'Протеин',
    3: 'Сладости'
}

for row in df['Класс']:
    colors.append(dict_colors.get(row))
plt.subplot(1, 2, 1)
plt.axis([0, 10, 0, 10])
plt.scatter(sladost_x, xpyct_y, c=colors)
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title('Изначальная выборка')
# plt.show()
# endregion
# region Заполнение массива с элементами выборки
learn_samples = []
for row in df['Сладость']:
    learn_samples.append([row, 0, 0])
i = 0
for row in df['Хруст']:
    learn_samples[i][1] = row
    i += 1
i = 0
for row in df['Класс']:
    learn_samples[i][2] = row
    i += 1

# endregion
# Вызовы методов для 3 классов
# Вызов метода для новых продуктов
new_samples = [[8, 2, 0], [2, 2, 2], [3, 8, 1]]
knn(learn_samples, new_samples, 5, 5)
# Вызов метода для тестовой выборки
test_samples = [[8, 2, 0], [2, 2, 2], [3, 8, 1], [7, 8, 3], [6, 8, 3]]
knn(learn_samples, test_samples, 5, 5, 'test')

# region Заполнение цсв файла для СК ЛЕРН (4 класса)
with open('Data.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\r')
    # 0 - Фрукт
    # 1 - Овощ
    # 2 - Протеин
    writer.writerow(['Продукт', 'Сладость', 'Хруст', 'Класс'])
    writer.writerow(['Яблоко', 9, 7, 0])
    writer.writerow(['Салат', 2, 5, 1])
    writer.writerow(['Бекон', 1, 2, 2])
    writer.writerow(['Банан', 9, 1, 0])
    writer.writerow(['Орехи', 1, 5, 2])
    writer.writerow(['Рыба', 1, 1, 2])
    writer.writerow(['Сыр', 1, 1, 2])
    writer.writerow(['Виноград', 8, 1, 0])
    writer.writerow(['Морковь', 2, 8, 1])
    writer.writerow(['Апельсин', 6, 1, 0])
    writer.writerow(['Леденец', 7, 9, 3])
    writer.writerow(['Шоколад', 9, 6, 3])
    writer.writerow(['Безе', 10, 10, 3])
    writer.writerow(['Вафля', 7, 8.5, 3])
    # test
    writer.writerow(['Мандарин', 8, 2, 0])
    writer.writerow(['Шашлык', 2, 2, 2])
    writer.writerow(['Капуста', 3, 8, 1])
    writer.writerow(['Казинак', 7, 8, 3])
    writer.writerow(['Печенье', 6, 8, 3])
# endregion
# region Чтение из цсв файла в датафрейм
df = pandas.read_csv('Data.csv', header=0, index_col=0, encoding='cp1251')
# endregion
# Вызов метода кнн склерн
knn_sklearn(5, df, 4)
