import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('configuration.ini')

file = config['display_new_dataset']['dataset_file']

file = 'tweets_dataset_textblob.csv'
df = pd.read_csv(file, index_col=None, lineterminator='\n', header=0, names= ['Datetime','Clean_Text','Text','Subjectivity','Polarity','Sentiment'], dtype='unicode')

print("The first 5 records")
print(df.head(5))
print("The last 5 records")
print(df.tail(5))
print("total number of records:")
print(len(df))
