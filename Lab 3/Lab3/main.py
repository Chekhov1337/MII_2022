import csv
import random
import statistics
import numpy
import pandas
import matplotlib.pyplot as plt

# Массивы для заполнения
surnames_male = ["Миридонов", "Евсеев", "Доронькин", "Дементьев", "Терентьев", "Альбертов", "Древесюк", "Амогусов"]
surnames_female = ["Миридонова", "Боброва", "Иванова", "Капибарова", "Беброва", "Хорькова", "Андреева", "Терентьева"]
posts = ["Начальник отдела", "Программист", "Администратор", "Технический специалист", "Сотрудник техподдержки"]

Data_list = []


# Статистические данные с помощью нумпай
def numpy_stat(column, csv_file, headers):
    header = headers[column]
    data = []

    for cell in csv_file:
        data.append(int(cell[column]))

    stat_array = numpy.array(data)
    print("_______________________NumPY___________________________")
    print("Статистические данные для столбца \"%s\" " % header)
    print("Минимальное значение:", numpy.min(stat_array))
    print("Максимальное значение:", numpy.max(stat_array))
    print("Среднее значение:", numpy.average(stat_array))
    print("Дисперсия:", numpy.var(stat_array))
    print("Стандартное отклонение:", numpy.std(stat_array))
    print("Медиана:", numpy.median(stat_array))
    print("Мода:", statistics.mode(stat_array))


# Статистические данные с помощью пандас
def pandas_stat(column, dataframe):
    print("_______________________Pandas___________________________")
    print("Статистические данные для столбца \"%s\" " % column)
    print("Минимальное значение:", dataframe[column].min())
    print("Максимальное значение:", dataframe[column].max())
    print("Среднее значение:", dataframe[column].mean())
    print("Дисперсия:", dataframe[column].var())
    print("Стандартное отклонение:", dataframe[column].std())
    print("Медиана:", dataframe[column].median())
    print("Мода:", dataframe[column].mode())


# Заполнение цсв файла
with open("Data.csv", "w") as f:
    writer = csv.writer(f, delimiter=",", lineterminator="\r")
    writer.writerow(
        ["Табельный номер", "Фамилия", "Пол", "Год рождения", "Год начала работы в компании", "Подразделение",
         "Должность", "Оклад", "Кол-во выполненных проектов"])
    for i in range(1, 1100):
        temp_sex = random.randint(0, 1)
        temp_birthyear = random.randint(1970, 1990)
        temp_startwork = random.randint(temp_birthyear + random.randint(18, 31), 2022)
        temp_post = random.choice(posts)
        temp_subdiv = random.randint(1, 5)
        ye_of_work = 2022 - temp_startwork
        if ye_of_work < 5:
            temp_projects = random.randint(1, 30)
        elif ye_of_work > 5:
            temp_projects = random.randint(30, 100)

        if ye_of_work == 1:
            temp_salary = random.randrange(20000, 25000, 1000) + temp_projects * 100
        elif ye_of_work < 5:
            temp_salary = random.randrange(20000, 25000, 1000) + temp_projects * 100
        elif ye_of_work < 7:
            temp_salary = random.randrange(25000, 30000, 1000) + temp_projects * 100
        elif ye_of_work > 7:
            temp_salary = random.randrange(30000, 35000, 1000) + temp_projects * 100
        temp_id = i
        if temp_sex == 0:
            sex_str = "Мужской"
            temp_surname = random.choice(surnames_male)
        elif temp_sex == 1:
            sex_str = "Женский"
            temp_surname = random.choice(surnames_female)
        writer.writerow(
            [temp_id, temp_surname, sex_str, temp_birthyear, temp_startwork, temp_subdiv,
             temp_post, temp_salary, temp_projects])

new_list = []
with open("Data.csv", "r") as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        new_list.append(row)

numpy_stat(3, new_list, headers)
numpy_stat(8, new_list, headers)
numpy_stat(7, new_list, headers)

df_stat = pandas.read_csv("Data.csv", header=0, index_col=0, encoding="cp1251")

pandas_stat("Год рождения", df_stat)
pandas_stat("Оклад", df_stat)
pandas_stat("Кол-во выполненных проектов", df_stat)

# Вывод графика зависимости Оклада от года начала работы в компании
plt.figure()
plt.subplot(2, 2, 1)
plt.title("Оклад")
plt.xlabel("Год начала работы в компании")
plt.ylabel("Оклад")
plt.plot(df_stat["Год начала работы в компании"].sort_values(), df_stat["Оклад"].sort_values())
# plt.show()
# Вывод графика зависимости Оклада от года начала работы в компании
plt.subplot(2, 2, 2)
plt.title("Оклад")
plt.xlabel("Количество выполненных проектов")
plt.ylabel("Оклад")
plt.plot(df_stat["Кол-во выполненных проектов"].sort_values(), df_stat["Оклад"].sort_values())
# plt.show()
# Построение круговой диаграммы
plt.subplot(2, 2, 3)
plt.title("Пол сотрудников")
labels = "Женщины", "Мужчины"
male_count = 0
female_count = 0
for sex in df_stat["Пол"]:
    if sex == "Мужской":
        male_count = male_count + 1
    elif sex == "Женский":
        female_count = female_count + 1
sizes = [female_count, male_count]
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

# Построение графика столбиками
plt.subplot(2,2,4)
plt.bar(labels, sizes)
plt.show()