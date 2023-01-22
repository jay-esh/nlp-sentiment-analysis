# nlp-sentiment-analysis
A small sentiment analysis project on AMZN stock using natural language processing

Here I used a pre-trained nlp model to predict the stock price: [https://huggingface.co/yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)

I used the matplot library to plot the actual and the prediction graph of the AMZN stock.
![prediction graph](https://github.com/[jay-esh]/[nlp-sentiment-analysis]/blob/[main]/Prediction.png?raw=true)
Orange curve is the predicted curve and the Blue curve is formed using the stock price data from yahoo finance. 

Used yahoo finance library to get the actual stock price data from the past.
I also used `pandas` to store the data in a DataFrame type as it makes it easier to work with the large time-stamped data.

