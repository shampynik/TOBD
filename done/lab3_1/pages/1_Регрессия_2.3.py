"""Страница 1 (самостоятельная работа): сравнение моделей регрессии из работы 2.3 (LazyRegressor, Rural_prices.csv)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Регрессия (работа 2.3)', layout='wide')

# Пути только для Windows: ...\done\lab_3_1\pages\<файл>.py  ->  ...\done\lab_3_1  ->  ...\done
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PAGES_DIR)
DONE_DIR = os.path.dirname(APP_DIR)
FILE_DATA = DONE_DIR + '\\Rural_prices.csv'


@st.cache_data
def load_data():
    return pd.read_csv(FILE_DATA, sep=';', decimal=',')


@st.cache_data(show_spinner='Строятся модели регрессии, это занимает от нескольких секунд до минуты...')
def build_models(test_size, seed):
    from lazypredict.Supervised import LazyRegressor
    data = load_data()
    x1 = data.drop('sale price', axis=1)
    y1 = data['sale price']
    X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=test_size, random_state=seed)
    clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models_train, _ = clf.fit(X_train, X_train, Y_train, Y_train)   # оценка на обучающей выборке (как в методичке)
    models_test, _ = clf.fit(X_train, X_test, Y_train, Y_test)      # оценка на тестовой выборке
    return models_train, models_test, len(X_train), len(X_test)


st.title('Сравнение моделей регрессии (работа 2.3)')
st.write('Цена продажи объекта недвижимости (`sale price`) по 11 характеристикам. '
         'Библиотека lazypredict строит все доступные модели регрессии с параметрами по умолчанию и сравнивает их.')

try:
    data = load_data()
except FileNotFoundError:
    st.error('Не найден файл ' + FILE_DATA)
    st.stop()

with st.expander('Данные Rural_prices.csv'):
    st.write('Наблюдений: %d, столбцов: %d (11 признаков и целевая переменная sale price)' % data.shape)
    st.dataframe(data.head(10))
    st.dataframe(data.describe().round(2))

test_size = st.sidebar.slider('Доля тестовой выборки', 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input('random_state разбиения', min_value=0, value=42, step=1)

if st.button('Построить модели', key='reg_build'):
    st.session_state['reg_built'] = True
if not st.session_state.get('reg_built'):
    st.info('Задайте параметры разбиения в боковой панели и нажмите «Построить модели».')
    st.stop()

try:
    models_train, models_test, n_train, n_test = build_models(float(test_size), int(seed))
except ImportError:
    st.error('Не установлена библиотека lazypredict (pip install lazypredict).')
    st.stop()

st.subheader('Результаты')
st.write('Обучающая выборка: %d объектов, тестовая: %d, построено моделей: %d' % (n_train, n_test, len(models_test)))

best = models_test['R-Squared'].idxmax()
st.success('Лучшая модель на тестовой выборке по R²: %s (R² = %.3f, RMSE = %.0f)' % (
    best, models_test.loc[best, 'R-Squared'], models_test.loc[best, 'RMSE']))

cmp = pd.DataFrame({
    'R2 train': models_train['R-Squared'],
    'R2 test': models_test['R-Squared'],
    'RMSE train': models_train['RMSE'],
    'RMSE test': models_test['RMSE'],
})
cmp['R2 train - test'] = cmp['R2 train'] - cmp['R2 test']
st.subheader('Сравнение обучающей и тестовой выборок')
st.caption('Большая разница R² на обучающей и тестовой выборках — признак переобучения.')
st.dataframe(cmp.sort_values('R2 test', ascending=False).round(3))

st.subheader('Диаграмма показателя')
col1, col2, col3 = st.columns(3)
sample = col1.radio('Выборка', ['Тестовая', 'Обучающая'], key='reg_sample')
metric = col2.selectbox('Показатель', ['R-Squared', 'RMSE', 'Time Taken'], key='reg_metric')
rmse_limit = col3.number_input('Предел шкалы RMSE', min_value=1000, value=75000, step=5000, key='reg_rmse_limit')

table = models_test if sample == 'Тестовая' else models_train

sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(6, 10))
sns.barplot(y=table.index, x=metric, data=table, ax=ax)
if metric == 'R-Squared':
    ax.set(xlim=(0, 1))
elif metric == 'RMSE':
    ax.set(xlim=(0, rmse_limit))
st.pyplot(fig)
plt.close(fig)

if metric == 'R-Squared':
    negative = table.index[table['R-Squared'] < 0].tolist()
    st.caption('Модели с отрицательным R² на диаграмме не видны (ось начинается с нуля): ' + (', '.join(negative) or 'нет'))
elif metric == 'RMSE':
    too_big = table.index[table['RMSE'] > rmse_limit].tolist()
    st.caption('Модели с RMSE выше предела шкалы: ' + (', '.join(too_big) or 'нет'))

st.subheader('Полная таблица метрик')
st.dataframe(table.round(4))
