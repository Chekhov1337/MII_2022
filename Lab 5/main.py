from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

learnSamplesSize = 0.3


def knnClassifier(dataframe, _nNeighbors=5, learnSamplesSize=0.2):
    x = dataframe.iloc[:, 0:9]
    y = dataframe.iloc[:, 9:]

    x_train, x_test, y_train, y_test = train_test_split(x, y.values.ravel(), train_size=learnSamplesSize)

    kNClassifier = KNeighborsClassifier(weights='distance', n_neighbors=_nNeighbors)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    kNClassifier.fit(x_train, y_train)

    yPrediction = kNClassifier.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, yPrediction)
    classificationReport = classification_report(y_test, yPrediction)

    return [kNClassifier.score(x_test, y_test), confusionMatrix, classificationReport]


def randomForesClassifier(dataframe, learnSamplesSize=0.2):
    x = dataframe.iloc[:, 0:9]
    y = dataframe.iloc[:, 9:]

    x_train, x_test, y_train, y_test = train_test_split(x, y.values.ravel(), train_size=learnSamplesSize)

    randForestClassifier = RandomForestClassifier()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    randForestClassifier.fit(x_train, y_train)

    yPrediction = randForestClassifier.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, yPrediction)
    classificationReport = classification_report(y_test, yPrediction)

    return [randForestClassifier.score(x_test, y_test), confusionMatrix, classificationReport]


def naiveBayesClassifier(dataframe, learnSamplesSize=0.2):
    x = dataframe.iloc[:, 0:9]
    y = dataframe.iloc[:, 9:]

    x_train, x_test, y_train, y_test = train_test_split(x, y.values.ravel(), train_size=learnSamplesSize)

    NBClassifier = GaussianNB()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    NBClassifier.fit(x_train, y_train)

    yPrediction = NBClassifier.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, yPrediction)
    classificationReport = classification_report(y_test, yPrediction)

    return [NBClassifier.score(x_test, y_test), confusionMatrix, classificationReport]


def showDataFrameFull(dataframe):
    print('Full DataSet:')
    print(tabulate(dataframe, tablefmt='github', headers='keys'))


def classifiersStat(dataframe, learnSamplesSize, iterationCount):
    knnScore = 0
    rtScore = 0
    nbScore = 0
    for i in range(0, iterationCount):
        knnScore += knnClassifier(dataframe, learnSamplesSize=learnSamplesSize)[0]
        rtScore += randomForesClassifier(dataframe, learnSamplesSize=learnSamplesSize)[0]
        nbScore += naiveBayesClassifier(dataframe, learnSamplesSize=learnSamplesSize)[0]

    fig, ax = plt.subplots()

    ax.bar([1, 2, 3], [knnScore / iterationCount, rtScore / iterationCount, nbScore / iterationCount],
           color=['purple', 'orange', 'green'])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['KNN', 'Random Tree', 'Naive Bayes'])
    fig.set_figwidth(10)
    fig.set_figheight(6)
    plt.title(f'Средняя точность моделей за {iterationCount} прогонов')
    plt.show()


####################Чтение из фала + очистка#######################
df = pd.read_csv('water_potability.csv')
df.dropna(inplace=True)
#####################КНН########################
result = knnClassifier(df, learnSamplesSize=learnSamplesSize)
confusionMatrix = result[1]
classificationReport = result[2]
print('___________________________________________')
print('Классификатор КНН:')
print(confusionMatrix)
print('True positive: ', confusionMatrix[1][1])
print('True negative: ', confusionMatrix[0][0])
print('False positive: ', confusionMatrix[0][1])
print('False negative: ', confusionMatrix[1][0])
print(classificationReport)
print('Точность модели: ', result[0])
disp = ConfusionMatrixDisplay(confusionMatrix)
disp.plot()
plt.title('Confusion matrix KNN')
plt.show()
##################Случайный лес#######################
result = randomForesClassifier(df, learnSamplesSize=learnSamplesSize)
confusionMatrix = result[1]
classificationReport = result[2]
print('___________________________________________')
print('Классификатор Random Forest:')
print(confusionMatrix)
print('True positive: ', confusionMatrix[1][1])
print('True negative: ', confusionMatrix[0][0])
print('False positive: ', confusionMatrix[0][1])
print('False negative: ', confusionMatrix[1][0])
print(classificationReport)
print('Точность модели: ', result[0])

disp = ConfusionMatrixDisplay(confusionMatrix)
disp.plot()
plt.title('Confusion matrix Random Forest')
plt.show()
###################Наивный байес######################
result = naiveBayesClassifier(df, learnSamplesSize=learnSamplesSize)
confusionMatrix = result[1]
classificationReport = result[2]
print('___________________________________________')
print('Классификатор Naive Bayes:')
print(confusionMatrix)
print('True positive: ', confusionMatrix[1][1])
print('True negative: ', confusionMatrix[0][0])
print('False positive: ', confusionMatrix[0][1])
print('False negative: ', confusionMatrix[1][0])

print(classificationReport)
print('Точность модели: ', result[0])

disp = ConfusionMatrixDisplay(confusionMatrix)
disp.plot()
plt.title('Confusion matrix Naive Bayes')
plt.show()

classifiersStat(df, learnSamplesSize=learnSamplesSize, iterationCount=2)
