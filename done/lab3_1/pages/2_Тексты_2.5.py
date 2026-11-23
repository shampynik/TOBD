"""Страница 2 (самостоятельная работа): классификатор новостных текстов из работы 2.5 (TF-IDF + KNN).

Предобработка текста и ответ модели берутся из telegram_bot.py (prepare_text, predict_rubric),
как это делается в ноутбуке работы 2.5.
"""
import asyncio
import os
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Классификация текстов (работа 2.5)')

# Пути только для Windows: ...\done\lab_3_1\pages\<файл>.py  ->  ...\done\lab_3_1  ->  ...\done
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PAGES_DIR)
DONE_DIR = os.path.dirname(APP_DIR)
LAB25_DIR = DONE_DIR + '\\lab_2_5'

EXAMPLES = {
    'Бизнес': 'Компания объявила о снижении выручки и сообщила, что акционеры одобрили сделку по продаже бизнеса. Банк повысил ставку по кредитам, а инвесторы ожидают роста цен на нефть и падения курса рубля.',
    'Культура': 'В музее открылась выставка современных художников, на которой представлены картины и скульптуры. Режиссёр представил новый спектакль, а театр объявил о премьере и фестивале искусства.',
    'Спорт': 'Футбольная команда выиграла матч у соперника со счётом два один и вышла в финал чемпионата. Тренер похвалил игроков, а нападающий забил победный гол на последней минуте.',
}


@st.cache_resource(show_spinner='Загружаются модели (natasha, TF-IDF, KNN), это занимает до минуты...')
def load_bot():
    """Импорт telegram_bot из папки lab_2_5 (бот при импорте не запускается: main() вызывается только при запуске файла)."""
    # Telethon при создании клиента ищет цикл событий; в потоке Streamlit его может не быть
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    if LAB25_DIR not in sys.path:
        sys.path.insert(0, LAB25_DIR)
    old_cwd = os.getcwd()
    os.chdir(LAB25_DIR)    # клиент Telethon создаёт файл сессии test_bot.session в текущей папке; пусть это будет lab_2_5
    try:
        import telegram_bot
    finally:
        os.chdir(old_cwd)
    return telegram_bot


st.title('Классификация новостных текстов (работа 2.5)')
st.write('Модель определяет рубрику новостного текста: очистка, удаление стоп-слов, именованных сущностей и чисел, '
         'лемматизация (natasha), векторизация TF-IDF и классификация методом k ближайших соседей.')

try:
    tb = load_bot()
except Exception as e:
    st.error('Не удалось загрузить telegram_bot.py из папки ' + LAB25_DIR + ': ' + repr(e))
    st.caption('Нужны: knn_text_classifier.joblib, DATA\\stop-words.txt и SECRET\\*.txt в папке lab_2_5, '
               'а также библиотеки natasha, telethon, joblib.')
    st.stop()

st.sidebar.header('О модели')
st.sidebar.write('Рубрики: ' + ', '.join(tb.rubrics))
st.sidebar.write('Число соседей: %d' % tb.model.n_neighbors)
st.sidebar.write('Слов в словаре TF-IDF: %d' % len(tb.vectorizer.get_feature_names_out()))

st.session_state.setdefault('news_text', '')


def set_example(text):
    st.session_state['news_text'] = text


st.write('Примеры текстов:')
columns = st.columns(len(EXAMPLES))
for (name, text), column in zip(EXAMPLES.items(), columns):
    column.button(name, on_click=set_example, args=(text,), key='example_' + name)

news_text = st.text_area('Текст новости (лучше 50 слов и больше)', key='news_text', height=220)

if st.button('Определить рубрику', key='classify'):
    if len(news_text.split()) < 5:
        st.warning('Текст слишком короткий: введите хотя бы несколько предложений.')
    else:
        with st.spinner('Обработка текста...'):
            name, probs = tb.predict_rubric(news_text)
        if name is None:
            st.warning('В тексте нет слов, известных модели, определить рубрику не удалось.')
        else:
            st.success('Рубрика: ' + name)
            st.subheader('Доли голосов ближайших соседей')
            shares = pd.Series(probs, name='доля голосов')
            st.bar_chart(shares)
            st.dataframe(shares.to_frame().round(2))
