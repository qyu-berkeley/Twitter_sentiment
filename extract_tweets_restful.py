from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
from pandas.io.json import json_normalize
from kafka import KafkaProducer
import urllib2

#producer = KafkaProducer(bootstrap_servers='localhost:9092')
#producer.send('twitterstream', b'some_message_bytes')

##########################################################################

num_of_tweets=1000 #enter number of tweets x 100 that need to be extracted

#### e.g. value of 3 would return max of 300 tweets
##########################################################################

# update api url
data = json.load(urllib2.urlopen("http://127.0.0.1:5000/movies"))

movienames= data['moviename'][0]
print movienames
movienames= ' OR '.join(movienames)
#### sample data -
##movienames='#moana OR #doctorstrange OR #allied OR #arrivalmovie OR #badsanta2 OR #almostchristmasmovie OR #assassinscreed  OR #collateralbeauty  OR #fantasticbeastsandwheretofindthem  OR #jackie  OR #lalaland  OR #passengers  OR #rogueonestarwarsstory  OR #sing'

#-----------------------------------------------------------------------
# load  API credentials
#-----------------------------------------------------------------------
config = {}
execfile("config.py", config)

#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(
		        auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))

#create a producer to write json messages to kafka
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))

#iterator=twitter.search.tweets(q='#KevinHartWhatNow', result_type='recent', lang='en', count=10)
f = open('/home/w205user/data/myfile_restful.dat','w')

i=0
iterator=twitter.search.tweets(q = movienames, result_type='recent', lang='en', count = 100)
n_max = float('+inf')
n_min = float('-inf')

for i in range(num_of_tweets):
	print "Search complete (%.3f seconds)" % (iterator["search_metadata"]["completed_in"])
	#json_normalize(iterator)[["text"]]
	count=0
	for tweet in iterator["statuses"]:
		count += 1
		#jsontweet = json.loads(tweet)
		tweet_id=tweet['id']
		#print tweet['id']
		if tweet_id < n_max:
			min_id=tweet_id
			n_max = tweet_id
		if tweet_id > n_min:
			max_id=tweet_id
			n_min = tweet_id

		producer.send('Twitter', tweet)
        json.dump(tweet, f)
        f.write('\n')
		###print json.dumps(tweet, indent=4)
	#print "#####################Minimum ID#################### %s" % min_id
	i += 1
	print count
	iterator=twitter.search.tweets(q = movienames, result_type='recent', lang='en', count = 100, max_id=min_id)
