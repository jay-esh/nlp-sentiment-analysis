import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import random
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datetime import date, timedelta
import torch

# pre-trained model for Natural Language processing on finance news
finbert = BertForSequenceClassification.from_pretrained(
    'yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# AMZN stock analysis from 2015 till 2017
# data taken from yahoo finance API
amzn = yf.Ticker("AMZN")
data = yf.download("AMZN", start="2015-01-01", end="2017-01-01")['Adj Close']

# got the news headlines from kaggle in a csv file
# reading a csv file
# storing the data in the form of a table - DataFrame type from pandas
df = pd.read_csv('us_equities_news_dataset.csv')

# changing to an appropriate date type
df['release_date'] = pd.to_datetime(df['release_date'])

# querying news headlines from the csv file between the time frame chosen
mask = (df['release_date'] > '2015-01-01') & (df['release_date'] <= '2017-01-01')
df = df.loc[mask]

# from the news headlines between the given dates we query the headlines for AMZN
df = df.query("ticker=='AMZN'")

# sorting them datewise in an ascending order
df = df.sort_values(by='release_date', ascending=True)
print(df)

# storing the sentiments of the headlines in a list (will be using this later)
sentiments = []
for i in df['title'].values:
    sentiments.append(nlp(i)[0]['label'])

# helper function to iterate over range of dates


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


start_date = date(2015, 1, 1)
end_date = date(2017, 1, 2)

plt.plot(data)

# creating a new column to the data table
df["Sentiment"] = sentiments

# initializing the values for graphing
price = data[0]
prices = []
# prices.append(price)
dates = []

for single_date in daterange(start_date, end_date):
    newmask = df['release_date'] == single_date.strftime("%Y-%m-%d")
    newdf = df.loc[newmask]
    # there are some dates missing since there are no news headlines for some days
    if newdf.empty == False:
        print(newdf['Sentiment'])
        for sentiment in newdf['Sentiment']:
            if sentiment == "Positive":
                price += 1
            elif sentiment == "Negative":
                price -= 1
            else:
                continue
        dates.append(single_date)
        prices.append(price)

# df["real-price"] = data
# print(df)
# print(prices)
# print(dates)
plt.show()
plt.plot(data)
plt.plot(dates, prices)
plt.show()
