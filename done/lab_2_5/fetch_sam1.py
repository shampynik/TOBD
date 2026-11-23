import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))     # папка, где лежит сам скрипт
SRC = os.path.join(BASE, "DATA", 'lenta-ru-news.csv.bz2')
OUT = os.path.join(BASE, "DATA", 'Data_selfwork')

TOPICS = ['Наука и техника', 'Путешествия', 'Силовые структуры']   # рубрики из столбца topic
N = 20                            # статей на рубрику (минимум 15)
MIN_WORDS, MAX_WORDS = 50, 150    # длина статьи в словах, как в Data.zip

head = pd.read_csv(SRC, nrows=5)
print('Столбцы:', list(head.columns))
text_col = 'text' if 'text' in head.columns else 'news'

pool = {t: [] for t in TOPICS}
seen_topics = set()
for chunk in pd.read_csv(SRC, usecols=['topic', text_col], chunksize=50000, low_memory=False):
    chunk = chunk.dropna()
    seen_topics.update(chunk['topic'].unique())
    for t in TOPICS:
        part = chunk.loc[chunk['topic'] == t, text_col]
        words = part.str.split().str.len()
        pool[t] += part[(words >= MIN_WORDS) & (words <= MAX_WORDS)].tolist()

missing = [t for t in TOPICS if not pool[t]]
if missing:
    raise SystemExit('Не найдены рубрики: %s\nДоступные рубрики: %s' % (missing, sorted(seen_topics)))

for t in TOPICS:
    sample = pd.Series(pool[t]).sample(n=min(N, len(pool[t])), random_state=42).tolist()
    folder = os.path.join(OUT, 'Рубрика ' + t.lower())
    os.makedirs(folder, exist_ok=True)
    for i, text in enumerate(sample, 1):
        with open(os.path.join(folder, '%d.txt' % i), 'w', encoding='utf-8') as f:
            f.write(text.strip())
    print(t, '-> статей:', len(sample), '| кандидатов было:', len(pool[t]))