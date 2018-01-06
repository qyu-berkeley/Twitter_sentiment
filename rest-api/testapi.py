import json
import urllib2
data = json.load(urllib2.urlopen("http://127.0.0.1:5000/movies"))
tempmovienames= data['moviename'][0]

movienames= ' OR '.join(tempmovienames)
print movienames
