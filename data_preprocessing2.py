import pandas as pd
import datetime
import pytz
import configparser
from function import cleanText
from function import sentiment_score_analysis, sentiment_type_analysis
from function import get_subjectivity, get_polarity, get_sentiment

config = configparser.ConfigParser()
config.read('configuration.ini')

tweets_raw_file = config['data_preprocessing2']['tweets_raw_file']
tweets_clean_file_path = config['data_preprocessing2']['tweets_clean_file_path']
time_zone = config['data_preprocessing2']['time_zone']
positive_boundary = float(config['data_preprocessing2']['positive_boundary'])
negative_boundary = float(config['data_preprocessing2']['negative_boundary'])
method = config['data_preprocessing']['method']

current_time = datetime.datetime.now(pytz.timezone(time_zone))
print('starting time: ' + str(current_time))
current_time_string = current_time.strftime("%Y%m%d-%H%M%S")

tweets_clean_file = tweets_clean_file_path + '/tweets_' + current_time_string + '.csv'
# print(tweets_clean_file)

df = pd.read_csv(tweets_raw_file,sep=';', index_col=None, header=0, names = ['Id', 'Datetime', 'Text'], usecols= ['Id', 'Datetime', 'Text'])
# print(df.head(5))

# clean up the tweets text
df['Clean_Text'] = df['Text'].apply(cleanText, args=(method,))
# print(method)

if (method == 'nltk'):
    df[['Neg','Neu','Pos','Compound']] = df['Text'].apply(sentiment_score_analysis).apply(pd.Series)
    df['Sentiment']= df['Compound'].apply(sentiment_type_analysis, args=(positive_boundary, negative_boundary))
    df.to_csv(tweets_clean_file, encoding='utf-8', mode='a', index=False, header=True, columns=['Datetime','Clean_Text','Text','Neg','Neu','Pos','Compound','Sentiment'])
    fig = df.Sentiment.value_counts().plot(kind='bar', title="sentiment analysis", figsize=(10, 10), fontsize=10).get_figure()
    fig.savefig(tweets_clean_file_path + '/Sentiment_Analysis_NLTK_' + current_time_string + '.jpg')

if (method == 'textblob'):
    df['Subjectivity']=df['Text'].apply(get_subjectivity)
    df['Polarity']=df['Text'].apply(get_polarity)    
    df['Sentiment']=df['Polarity'].apply(get_sentiment, args=(positive_boundary, negative_boundary))
    df.to_csv(tweets_clean_file, encoding='utf-8', mode='a', index=False, header=True, columns=['Datetime','Clean_Text','Text','Subjectivity','Polarity','Sentiment'])
    fig = df.Sentiment.value_counts().plot(kind='bar', title="sentiment analysis", figsize=(10, 10), fontsize=10).get_figure()
    fig.savefig(tweets_clean_file_path + '/Sentiment_Analysis_TextBlob_' + current_time_string + '.jpg')

current_time = datetime.datetime.now(pytz.timezone(time_zone))
print('ending time: ' + str(current_time))
