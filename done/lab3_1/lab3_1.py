"""Лабораторная работа 3.1. Главная страница: классификация ирисов (основная часть методички).

Запуск (Windows, из папки done\\lab_3_1):  python -m streamlit run lab3_1.py
Страницы с моделями из работ 2.3, 2.5 и 2.6 (самостоятельная работа) лежат в папке pages
и появляются в меню слева автоматически.
"""
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Классификация цветов')

st.write("""
# Классификация цветов
Приложение классифицирует вид ириса по выбранным параметрам
""")
st.sidebar.header('Параметры для классификации')


def user_input_features():
    sepal_length = st.sidebar.slider('Длина чашелистика', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ширина чашелистика', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Длина лепестка', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('Выбранные параметры пользователем')
st.write(df)

iris = datasets.load_iris()
# Признаки передаются таблицей с теми же названиями столбцов, что и у df (иначе sklearn выдаёт предупреждение)
X = pd.DataFrame(iris.data, columns=df.columns)
Y = iris.target


# Модель обучается один раз и хранится в кэше: Streamlit выполняет скрипт заново при каждом движении ползунка
@st.cache_resource
def train_model():
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, Y)
    return clf


clf = train_model()

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Возможная классификация и индекс вида')
st.write(iris.target_names)

st.subheader('Прогноз')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Вероятность прогноза')
st.write(prediction_proba)

st.sidebar.caption('Модели из работ 2.3, 2.5 и 2.6 — на страницах в меню выше.')
