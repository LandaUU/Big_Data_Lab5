import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import re

def is_drop_row(df, ind, number_to_drop):
    # Скорей всего есть более красивый и оптимальный способ, но сделал как смог :)
    null_count = 0
    if pd.isnull(df.loc[ind, 'city']) or pd.isna(df.loc[ind, 'city']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'company_name']) or pd.isna(df.loc[ind, 'company_name']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'description']) or pd.isna(df.loc[ind, 'description']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'duty']) or pd.isna(df.loc[ind, 'duty']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'employment']) or pd.isna(df.loc[ind, 'employment']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'experience']) or pd.isna(df.loc[ind, 'experience']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'max_salary']) or pd.isna(df.loc[ind, 'max_salary']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'min_salary']) or pd.isna(df.loc[ind, 'min_salary']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'name']) or pd.isna(df.loc[ind, 'name']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'published_date']) or pd.isna(df.loc[ind, 'published_date']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'requirements']) or pd.isna(df.loc[ind, 'requirements']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'schedule']) or pd.isna(df.loc[ind, 'schedule']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'skills']) or pd.isna(df.loc[ind, 'skills']):
        null_count = null_count + 1
    if pd.isnull(df.loc[ind, 'terms']) or pd.isna(df.loc[ind, 'terms']):
        null_count = null_count + 1
    if null_count >= number_to_drop:
        return [True, null_count]
    return [False, null_count]

def Chapter_1():
    df = pd.read_csv("Data/Moscow.csv")
    # ЧАСТЬ 1
    # Выполнить сокращение датасета с помощью сокращения размерности: удалить вакансии, у которых больше 70% пропусков по
    # признакам. 14 столбцов, это значит 14 * 0,7 ~ 9 признаков. Однако у меня не находятся строки я 9 пропусками,
    # поэтому чтобы было хоть что зачистить буду удалять строки с 7 и более пропущенными признаками.
    for ind in df.index:
        result = is_drop_row(df, ind, 7)
        if result[0]:
            df = df.drop(index=ind)
        print('is_drop_row result: ' + str(result[0]) + ', index: ' + str(ind) + ', null values count: ' + str(result[1]))

    df.to_csv('Data/Moscow_clean.csv')


def classife_name(df):
    # Группировка вакансий по названиям
    vacancy_list = ['разработчик|developer|программист', 'инженер|engineer', 'архитектор|architect', 'аналитик|analyst',
                    'data scientist', 'c++', 'c#', 'python', 'js|javascript', '1c', 'goland', 'frontend',
                    'backend', 'full-stack|fullstack', 'security|безопасность']

    # Видимо раз есть у меня группы, то просто по первая группа - 0 и т.д.

    df_group = pd.DataFrame()
    counter = 0
    dictionary = {}
    for vacancy in vacancy_list:
        for v in vacancy.split('|'):
            dfV = df[df["name"].replace('/', '').str.contains(re.escape(v))]
            dfV['name'] = counter
        if len(df_group.columns) == 0:
            df_group = dfV.copy()
        else:
            df_group.append(dfV)
        group_name = vacancy
        dictionary[group_name] = counter
        # df_group = pd.DataFrame()
        counter = counter + 1
    print(dictionary)
    return df_group

def encoder_column(column_name, df):
    le = LabelEncoder()
    le.fit(df[column_name])
    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_mapping)
    print(le.transform(df[column_name]))
    return le_mapping

def class_1(df):
    id = df.iloc[:, 0]
    X = df.iloc[:, 1:6]
    Y = df.iloc[:, 7]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=27)
    print(X_train)
    print(y_train)
    SVC_model = SVC()
    # В KNN-модели нужно указать параметр n_neighbors
    # Это число точек, на которое будет смотреть
    # классификатор, чтобы определить, к какому классу принадлежит новая точка
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    SVC_model.fit(X_train, y_train)
    KNN_model.fit(X_train, y_train)
    SVC_prediction = SVC_model.predict(X_test)
    KNN_prediction = KNN_model.predict(X_test)
    # Оценка точности — простейший вариант оценки работы классификатора
    print(accuracy_score(SVC_prediction, y_test))
    print(accuracy_score(KNN_prediction, y_test))
    # Но матрица неточности и отчёт о классификации дадут больше информации о производительности
    print(confusion_matrix(SVC_prediction, y_test))
    print(classification_report(KNN_prediction, y_test))


def Chapter_2():
    # company_name, description, duty, published_date, requirements, terms, skills - столбцы, которые нельзя просто
    # пропустить через labelEncoder
    df = pd.read_csv("Data/Moscow.csv")
    le = LabelEncoder()

    le.fit(df['experience'])
    le_experience_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_experience_mapping)
    # print(le.transform(df['experience']))
    df['experience'] = le.transform(df['experience'])
    # print(df['experience'])
    le.fit(df['employment'])
    le_employment_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_employment_mapping)
    # print(le.transform(df['employment']))
    df['employment'] = le.transform(df['employment'])

    le.fit(df['schedule'])
    le_schedule_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_schedule_mapping)
    # print(le.transform(df['schedule']))
    df['schedule'] = le.transform(df['schedule'])

    le.fit(df['city'])
    le_city_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_city_mapping)
    # print(le.transform(df['city']))
    df['city'] = le.transform(df['city'])

    # classife_name(df)

    df.drop(['company_name', 'description', 'duty', 'published_date', 'requirements', 'terms', 'skills'], axis=1, inplace=True)
    df = df.dropna()
    print(df)
    class_1(df)
Chapter_2()