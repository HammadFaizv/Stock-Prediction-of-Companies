#website libraries
from flask import Flask,request, url_for, redirect, render_template

#api libraries
import requests, json, urllib.request

#news processing libraries
from training_data import prediction,data_cleaner

#machine learning model libraries
import joblib
import numpy as np

#time input
import datetime

app = Flask(__name__)

model=joblib.load('StockLogReg.pkl')
date = datetime.datetime.now()
today_date = str(date.year)+'-'+str(date.month)+'-'+str(date.day)

def model_prediction(data):
    data = prediction(data)
    return model.predict_proba(data[:,:584289])

@app.route('/')
def hello_world():
    return render_template("stocks.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    stockName = [x for x in request.form.values()][0]
    #print("\n\n\n\n\n\n",stockName,"\n\n\n\n\n\n")
    stock = stockName.replace(" ","").lower()
    key = '7af6ce7078524ceeb098006f32cbdd95'
    url = 'https://newsapi.org/v2/everything?q='+ stock +'&from='+ today_date +'&sortBy=publishedAt&apiKey='+key
    url_req = urllib.request.urlopen(url)
    news = json.load(url_req)
    news_headlines = news.get("articles")
    headlines = []
    for i in range(0,len(news_headlines)):
        headlines.append(news_headlines[i].get("description"))

    train_data = data_cleaner(headlines)
    pred = model_prediction(train_data)
    #return render_template("forest.html",pred=pred[0][0])
    if (pred[0][1]>0.5):
        return render_template('stock_stonks.html',pred='The Stocks for {} are likely to go up.\nProbability of stock increasing is {}'.format(stockName,pred[0][1]))
    else:
        return render_template('stock_not_stonks.html',pred='The {} stocks doesnt seem to increase.\n Probability of stocks going up is {}'.format(stockName,pred[0][1]))



if __name__ == '__main__':
    app.run(debug=True)


#api : 7af6ce7078524ceeb098006f32cbdd95
