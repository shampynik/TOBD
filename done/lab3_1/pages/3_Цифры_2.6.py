"""Страница 3 (самостоятельная работа): распознавание цифр из работы 2.6 (GaussianNB, load_digits)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title='Распознавание цифр (работа 2.6)')


@st.cache_resource
def train_digits():
    numbers = load_digits()
    n_samples = len(numbers.images)
    X = numbers.images.reshape((n_samples, -1))
    Y = numbers.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    GB = GaussianNB()
    fit = GB.fit(X_train, y_train)
    _, idx_test = train_test_split(np.arange(n_samples), random_state=0)    # номера изображений тестовой выборки
    return numbers, X, Y, fit, X_test, y_test, set(idx_test.tolist())


numbers, X, Y, fit, X_test, y_test, idx_test = train_digits()
predicted_test = fit.predict(X_test)

st.title('Распознавание рукописных цифр (работа 2.6)')
st.write('Наивный байесовский классификатор (GaussianNB) на изображениях цифр 8×8 пикселей из набора load_digits '
         '(1797 изображений, разбиение 75/25, random_state=0).')

col1, col2 = st.columns(2)
col1.metric('Accuracy на тестовой выборке', '%.3f' % accuracy_score(y_test, predicted_test))
col2.metric('Изображений в тестовой выборке', len(y_test))

with st.expander('Матрица несоответствий на тестовой выборке'):
    cm = confusion_matrix(y_test, predicted_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Ответ модели')
    ax.set_ylabel('Реальная цифра')
    st.pyplot(fig)
    plt.close(fig)

mode = st.radio('Что показать', ['Одно изображение', 'Ряд цифр от 0 до 9 (самостоятельная работа 2.6)'], key='digits_mode')

if mode == 'Одно изображение':
    index = st.slider('Номер изображения', 0, len(Y) - 1, 0, key='digits_index')
    probs = fit.predict_proba(X[index:index + 1])[0]
    answer = int(np.argmax(probs))
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(numbers.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.axis('off')
    left.pyplot(fig)
    plt.close(fig)
    right.write('Реальная цифра: **%d**' % Y[index])
    right.write('Ответ модели: **%d** (%s)' % (answer, 'верно' if answer == Y[index] else 'ошибка'))
    right.write('Изображение из %s выборки' % ('тестовой' if index in idx_test else 'обучающей'))
    st.subheader('Вероятности цифр')
    st.bar_chart(pd.Series(probs, index=[str(d) for d in range(10)], name='вероятность'))
else:
    assert list(numbers.target[:10]) == list(range(10))     # первые 10 изображений набора - цифры 0, 1, ..., 9
    predicted_row = fit.predict(X[:10])
    fig, axes = plt.subplots(2, 5, figsize=(10, 5.4))
    for digit, ax in enumerate(axes.ravel()):
        ok = predicted_row[digit] == Y[digit]
        ax.imshow(numbers.images[digit], cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Цифра %d: ответ %d' % (Y[digit], predicted_row[digit]), color='green' if ok else 'red')
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    table = pd.DataFrame({
        'Реальная цифра': Y[:10],
        'Ответ модели': predicted_row,
        'Результат': ['верно' if p == y else 'ошибка' for p, y in zip(predicted_row, Y[:10])],
        'Выборка': ['тестовая' if i in idx_test else 'обучающая' for i in range(10)],
    })
    st.dataframe(table)
    st.write('Верно распознано: %d из 10' % (predicted_row == Y[:10]).sum())
