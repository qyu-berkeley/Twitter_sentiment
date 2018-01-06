
import sys
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)


from kafka import KafkaConsumer
import pandas as pd
import json
import pandas as pd
from textblob import TextBlob
import nltk
import gensim
import spacy
import textprocessing
import re
from textprocessing import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
from textprocessing import textfeatures
import psycopg2
from django.utils.encoding import smart_str, smart_unicode
from datetime import datetime


def initialize(db_name = "tweetdata"):
    '''
    Initialize a function that takes postgres database name as arguent and sets up a pandas dataframe.
    Connection to postgres database is established using psycopg2 postgres connector and cursor is initiated to run
    SQL queries. Function returns, the tweets dataframe, postgres connector and the SQL cursor as objects.
    '''
    tweets = pd.DataFrame() # set-up an pandas dataframe
    # set-up a postgres connection
    conn = psycopg2.connect(database=db_name, user="postgres",password="pass",  host="localhost", port="5432")
    dbcur = conn.cursor()
    print "connection successful"
    return (tweets, conn, dbcur)


def extracttweetfeatures(tweets,output):
    '''
    extracttweetfeatures function takes tweets dataframe and a output list as input. Output list comprises of the list
    of all tweets in a json format consumed by json consumer. Function theb extracts the important features such as
    tweet text, movie name, language, country, user name, coordinates, location, retweets count.
    '''
    try:
        tweets['text'] = map(lambda tweet: tweet['text'], output)
    except IndexError:
        tweets['text'] = 'Data Error'
    try:
        tweets['movie'] = map(lambda tweet: tweet['entities']['hashtags'][0]['text'], output)
    except IndexError:
        tweets['movie'] = 'Data Error'
    try:
        tweets['lang'] = map(lambda tweet: tweet['user']['lang'], output)
    except IndexError:
        tweets['lang'] = 'Data Error'
    try:
        tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, output)
    except IndexError:
        tweets['country'] = 'Data Error'
    try:
        tweets['user_nm'] = map(lambda tweet: tweet['user']['name'].encode('utf-8'), output)
    except IndexError:
        tweets['user_nm'] = 'Data Error'

    try:
        tweets['screen_nm'] = map(lambda tweet: tweet['user']['screen_name'].encode('utf-8'), output)
    except IndexError:
        tweets['screen_nm'] = 'Data Error'
    try:
        tweets['coordinates_lat'] = map(lambda tweet: str(tweet['coordinates']['coordinates'][1]) if tweet['coordinates'] != None else None, output)
    except IndexError:
        tweets['coordinates_lat'] = 'Not available'
    except KeyError:
        tweets['coordinates_lat'] = 'Not available'
    except TypeError:
        tweets['coordinates_lat'] = 'Not available'
    try:
        tweets['coordinates_long'] = map(lambda tweet: str(tweet['coordinates']['coordinates'][0]) if tweet['coordinates'] != None else None , output)
    except IndexError:
        tweets['coordinates_long'] = 'Not available'
    except KeyError:
        tweets['coordinates_long'] = 'Not available'
    except TypeError:
        tweets['coordinates_long'] = 'Not available'

    try:
        tweets['location'] = map(lambda tweet: tweet['user']['location'] if tweet['user'] != None else None, output)
    except IndexError:
        tweets['location'] = 'Data Error'
    try:
        tweets['retweets_count'] = map(lambda tweet: tweet['retweeted_status']['retweet_count'], output)
    except IndexError:
        tweets['retweets_count'] = 0
    except KeyError:
        tweets['retweets_count'] = 0
    try:
        tweets['followers_count'] = map(lambda tweet: tweet['user']['followers_count'], output)
    except IndexError:
        tweets['followers_count'] = 0
    except KeyError:
        tweets['followers_count'] = 0
    try:
        tweets['favourites_count'] = map(lambda tweet: tweet['user']['favourites_count'], output)
    except IndexError:
        tweets['favourites_count'] = 0
    except KeyError:
        tweets['favourites_count'] = 0
    try:
        tweets['friends_count'] = map(lambda tweet: tweet['user']['friends_count'], output)
    except IndexError:
        tweets['friends_count'] = 0
    except KeyError:
        tweets['friends_count'] = 0


def cleantweettext(tweets):
    '''
    cleantweettext function takes tweets dataframe. Function adds a text_clean column to tweets dataframe by
    running text cleansing functions.
    '''
    tweets['text_clean'] = [re.sub(r"http\S+", "", v) for v in tweets.text.values.tolist()]
    tweets['text_clean'] = [re.sub(r"#\S+", "", v) for v in tweets.text_clean.values.tolist()]
    tweets['text_clean'] = [re.sub(r"@\S+", "", v) for v in tweets.text_clean.values.tolist()]
    tweets['text_clean'] = [re.sub(r"u'RT\S+", "", v) for v in tweets.text_clean.values.tolist()]
    #tweets['text'] = [v.replace('\n'," ") for v in tweets.text.values.tolist()]
    #tweets['text'] = [v.replace(u"\u2018", " ").replace(u"\u2019", " ") for v in tweets.text.values.tolist()]
    try:
       tweets['text'] = [smart_str(v) for v in tweets.text.values.tolist()]
    except UnicodeDecodeError:
       tweets['text'] = [v.decode('utf-8') for v in tweets.text.values.tolist()]
    try:
       tweets['user_nm'] = [smart_str(v) for v in tweets.user_nm.values.tolist()]
    except UnicodeDecodeError:
       tweets['user_nm'] = [v.decode('utf-8') for v in tweets.user_nm.values.tolist()]
    try:
       tweets['location'] = [smart_str(v) for v in tweets.location.values.tolist()]
    except UnicodeDecodeError:
       tweets['location'] = [v.decode('utf-8') for v in tweets.location.values.tolist()]
    tweets['text_clean'] = preprocessing.clean_text(text=tweets.text_clean.values, remove_short_tokens_flag=False ,lemmatize_flag=True)

def calculatesentiments(tweets):
    '''
    calculatesentiments function takes tweets dataframe. Function then uses vader lexicon to compute the sentiment
    scores for all the tweets. Further, the scores are then used to classify tweets as positive, negative and neutral.
    '''
    tweets['sentiment_score'] = [textfeatures.score_sentiment(v)['compound'] for v in tweets.text_clean.values.tolist()]
    tweets.loc[tweets['sentiment_score'] > 0.0, 'sentiment'] = 'positive'
    tweets.loc[tweets['sentiment_score'] == 0.0, 'sentiment'] = 'neutral'
    tweets.loc[tweets['sentiment_score'] < 0.0, 'sentiment'] = 'negative'


def cleanse_dataframe_and_load(tweets,conn, dbcur):
    '''
    cleanse_dataframe_and_load function takes tweets dataframe, postgres connector and cursor.
    Function dedupes the data frame for any duplicate tweets and then inserts the rows into the postgres database
    '''
    tweet_dedup = tweets.drop_duplicates(['text'], keep =False)
    data1 = [tuple(x) for x in tweet_dedup.to_records(index=False)][0]
    #print dat
    temp = list(data1)
    dt = datetime.now()
    temp.append(dt)
    data = tuple(temp)
    if data[0] == 'Data Error':
        pass
    else:
        querystr = dbcur.mogrify("INSERT INTO tweets VALUES (%s, %s, %s, %s, %s, %s, %s,%s, %s, %s,%s, %s, %s, %s, %s, %s,%s) ON CONFLICT DO NOTHING;", data)
        #print querystr
        dbcur.execute(querystr)
        conn.commit()


def inserttweetwords(tweets, conn, dbcur):
    print tweets['text_clean'][0]
    tweet_words = tweets['text_clean'][0].split(' ')
    print tweet_words
    for each in tweet_words:
        if len(each) <= 3:
            pass
        else:
            score = textfeatures.score_sentiment(each)['compound']
            movie_nm = tweets['movie'][0]
            word_sent = ["positive" if score > 0 else "neutral" if score == 0 else "negative"]
            querystr = dbcur.mogrify("INSERT INTO tweet_words as tw (word, movie, count, word_sentiment) VALUES (%s, %s, 1, %s) ON CONFLICT (word) DO UPDATE SET count = tw.count + 1 WHERE tw.word = %s and tw.movie = %s ;", (each, movie_nm, word_sent, each, movie_nm))
            #print querystr
            dbcur.execute(querystr)
            conn.commit()



def main():
    '''
    main function initiates a kafka consumer, initialize the tweetdata database. Consumer consumes tweets from producer
    extracts features, cleanses the tweet text, calculates sentiments and loads the data into postgres database
    '''
    consumer = KafkaConsumer('Twitter')  # set-up a Kafka consumer
    tweets,conn, dbcur = initialize(db_name = "tweetdata")
    for msg in consumer:
        output = []
        output.append(json.loads(msg.value))
        print output
        print '\n'
        extracttweetfeatures(tweets, output)
        cleantweettext(tweets)
        calculatesentiments(tweets)
        cleanse_dataframe_and_load(tweets, conn, dbcur)
        inserttweetwords(tweets, conn, dbcur)


if __name__ == "__main__":
    main()
