import pandas as pd
import datetime
import pytz
import glob
import configparser
from function import cleanText
from function import sentiment_score_analysis, sentiment_type_analysis
from function import get_subjectivity, get_polarity, get_sentiment

config = configparser.ConfigParser()
config.read('configuration.ini')

tweets_raw_file_path = config['data_preprocessing']['tweets_raw_file_path']
tweets_clean_file_path = config['data_preprocessing']['tweets_clean_file_path']
time_zone = config['data_preprocessing']['time_zone']
positive_boundary = float(config['data_preprocessing']['positive_boundary'])
negative_boundary = float(config['data_preprocessing']['negative_boundary'])
method = config['data_preprocessing']['method']


current_time = datetime.datetime.now(pytz.timezone(time_zone))
print('starting time: ' + str(current_time))
current_time_string = current_time.strftime("%Y%m%d-%H%M%S")

tweets_clean_file = tweets_clean_file_path + '/tweets_' + current_time_string + '.csv'
#print(tweets_clean_file)

tweets_raw_files  = glob.glob(tweets_raw_file_path + '/tweets_*.csv')
#print(tweets_raw_files)

df_list = []
for tweets_raw_file in tweets_raw_files:
    df = pd.read_csv(tweets_raw_file, index_col=None, header=0, usecols= ['Datetime', 'Tweet Id', 'Text', 'Username' ])
    df_list.append(df)

df = pd.concat(df_list, axis=0, ignore_index=True)
# print(df.head(5))
     
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
    print(df.head(5))
    df.to_csv(tweets_clean_file, encoding='utf-8', mode='a', index=False, header=True, columns=['Datetime','Clean_Text','Text','Subjectivity','Polarity','Sentiment'])
    fig = df.Sentiment.value_counts().plot(kind='bar', title="sentiment analysis", figsize=(10, 10), fontsize=10).get_figure()
    fig.savefig(tweets_clean_file_path + '/Sentiment_Analysis_TextBlob_' + current_time_string + '.jpg')

current_time = datetime.datetime.now(pytz.timezone(time_zone))
print('ending time: ' + str(current_time))
