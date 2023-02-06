import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re

df = pd.read_csv('Data/Data.csv')
words = set(nltk.corpus.words.words())

headlines = []
cleaned_df = df.copy()
cleaned_df.replace('[^a-zA-Z]', ' ',regex=True,inplace=True)
cleaned_df.replace('[ ]+', ' ',regex=True,inplace=True)
for row in range(len(df)):
    headlines.append(' '.join(str(x) for x in cleaned_df.iloc[row,2:]).lower())

cv = CountVectorizer(ngram_range=(2,2))
cv.fit(headlines)


def data_cleaner(headlines):
    lines = []
    for i in range(1,len(headlines)):
        line = re.sub('[^A-Za-z]',' ',headlines[i])
        line = re.sub('[ ]+', ' ', line)
        line = line.lower()
        lines.append(line)
    data = ' '.join(x for x in lines)
    data = " ".join(w for w in nltk.wordpunct_tokenize(data) if w.lower() in words or not w.isalpha())
    return data


def prediction(data):
    print("\n\n\ntransformation done!!!\n\n\n")
    return cv.transform([data])
