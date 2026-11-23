"""Телеграмм-бот на Telethon (MTProto через MTProxy): определяет рубрику новостного текста (TF-IDF + KNN)."""
import asyncio
import inspect
import io
import os
import re
import time; _t0 = time.time() 

import joblib

# Совместимость pymorphy2 (его использует natasha) с Python 3.11+
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, NewsSyntaxParser, Segmenter
from telethon import events
from telethon.sync import TelegramClient
from telethon.network.connection import ConnectionTcpMTProxyRandomizedIntermediate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_secret(name):
    with open(os.path.join(BASE_DIR, "SECRET", name), encoding='utf-8-sig') as f:
        return f.read().strip()


# ---------- Данные для подключения ----------
bot_token = read_secret('BOT_TOKEN.txt')
proxy_secret = read_secret('MT_PROXY_SECRET.txt')
api_data = read_secret('TELEGRAM_API.txt').splitlines()

api_id = int(api_data[0])
api_hash = api_data[1].strip()

# MTProto proxy
proxy = ('127.0.0.1', 1080, proxy_secret)

client = TelegramClient(
    'test_bot',
    api_id,
    api_hash,
    connection=ConnectionTcpMTProxyRandomizedIntermediate,
    proxy=proxy,
    timeout=10,
    connection_retries=1,
    request_retries=1,
    retry_delay=1,
    auto_reconnect=False
)

# ---------- Модель (создаётся в ноутбуке lab2.5.ipynb) ----------
artifacts = joblib.load(os.path.join(BASE_DIR, 'knn_text_classifier.joblib'))
vectorizer = artifacts['vectorizer']
model = artifacts['model']
rubrics = artifacts['rubrics']

with open(os.path.join(BASE_DIR, "DATA", 'stop-words.txt'), 'r', encoding='utf-8-sig') as f:
    stop_words = {line.strip().lower() for line in f if line.strip()}

# Модели natasha создаются один раз
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()
print('модели загружены', round(time.time() - _t0, 1), 'с')    # после строки morph_vocab = MorphVocab()

def clean_text(raw_text):
    """Простая очистка текста (как cleanLines в ноутбуке)."""
    text = ''
    for line in io.StringIO(raw_text).readlines():
        text += line
        text = text.replace('.', '')
        text = text.replace(',', '')
        text = text.replace(' — ', ' ')
        text = text.replace(' - ', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('  ', ' ')
    return text


def prepare_text(raw_text):
    """Три вида очистки (как getPreparedTexts в ноутбуке); возвращает текст из лемм."""
    words = re.findall(r'\b\S+\b', clean_text(raw_text))
    text = ' '.join(w for w in words if w.lower() not in stop_words)          # 1. стоп-слова

    doc = Doc(text)                                                           # 2. сущности и цифры
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        text = text.replace(span.text, '')
    text = text.replace('   ', ' ').replace('  ', ' ')
    text = ' '.join(w for w in re.findall(r'\b\S+\b', text) if not w.isdigit())

    doc = Doc(text)                                                           # 3. леммы
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return ' '.join({t.text: t.lemma for t in doc.tokens}.values())


def predict_rubric(raw_text):
    """Возвращает (рубрика, {рубрика: доля голосов соседей}) или (None, None), если слов из словаря нет."""
    vec = vectorizer.transform([prepare_text(raw_text)])
    if vec.nnz == 0:
        return None, None
    probs = model.predict_proba(vec)[0]
    return rubrics[int(model.predict(vec)[0])], dict(zip(rubrics, probs))


# ---------- Обработчик сообщений ----------
HELP_TEXT = 'Пришлите текст новости (лучше 50 слов и больше), и я определю рубрику: ' + ', '.join(rubrics) + '.'


@client.on(events.NewMessage(incoming=True))
async def on_message(event):
    text = event.raw_text or ''

    if text.startswith('/start') or text.startswith('/help'):
        await event.reply(HELP_TEXT)
        return
    if text.startswith('/'):
        return
    if len(text.split()) < 5:
        await event.reply('Текст слишком короткий: пришлите хотя бы несколько предложений.')
        return

    # Классификация выполняется в отдельном потоке, чтобы не блокировать соединение с Telegram
    loop = asyncio.get_running_loop()
    name, probs = await loop.run_in_executor(None, predict_rubric, text)
    if name is None:
        await event.reply('В тексте нет слов, известных модели, определить рубрику не удалось.')
        return

    shares = '\n'.join('{}: {:.0f}%'.format(r, p * 100) for r, p in probs.items())
    await event.reply('Рубрика: {}\n\nДоли голосов ближайших соседей:\n{}'.format(name, shares))


def main():
    # auto_reconnect=False: при обрыве связи клиент останавливается, поэтому подключаемся заново в цикле
    while True:
        try:
            print('подключаюсь...', round(time.time() - _t0, 1), 'с')      # в main(), перед client.start(...)
            client.start(bot_token=bot_token)
            print('Бот запущен')
            client.run_until_disconnected()
            print('Соединение разорвано')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('Ошибка соединения:', repr(e))
        time.sleep(5)
    client.disconnect()


if __name__ == '__main__':
    main()