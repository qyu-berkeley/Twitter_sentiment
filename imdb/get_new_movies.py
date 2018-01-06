from imdb import IMDb
import StringIO
import csv
import re
import psycopg2

out = StringIO.StringIO()
csv_file = 'movie_data.csv'
f1 = open(csv_file, 'w')
ia = IMDb('http')

# Getting 10 new movies from IMDB
movie1 = ia.get_movie('3748528')
movie2 = ia.get_movie('1211837')
movie3 = ia.get_movie('3183660')
movie4 = ia.get_movie('3521164')
movie5 = ia.get_movie('1619029')
movie6 = ia.get_movie('3783958')
movie7 = ia.get_movie('4682786')
movie8 = ia.get_movie('1355644')
movie9 = ia.get_movie('2094766')
movie10 = ia.get_movie('3470600')
movies = (movie1, movie2, movie3, movie4, movie5, movie6, movie7, movie8, movie9, movie10)

# set-up a postgres connection
db_name = "tweetdata"
conn = psycopg2.connect(database=db_name, user="postgres",password="pass",  host="localhost", port="5432")
cur = conn.cursor()
cur.execute('''CREATE TABLE movie_data
       (title TEXT,
       main_cast TEXT,
       director TEXT,
       genres TEXT,
       lang TEXT,
       plot TEXT,
       imdbURL TEXT
       );''')
conn.commit()
conn.close()

cw = csv.writer(out)
header = ("Title", "Cast", "Director", "Genre", "language", "Plot_Summary", "imdb_URL")
cw.writerow(header)
print header

for movie in movies:

    conn = psycopg2.connect(database=db_name, user="postgres",password="pass",  host="localhost", port="5432")
    cur = conn.cursor()

    imdbURL = ia.get_imdbURL(movie)
    if not imdbURL:
        imdbURL = 'NA'

    genres = movie.get('genres')
    if not genres:
        genres = 'NA'
    else:
        genres = [item.encode('utf-8') for item in genres]

    director = movie.get('director')
    if not director:
        director = 'NA'
    else:
        director = director[0]

    lang = movie.get('lang')
    if not lang:
        lang = 'NA'
    else:
        lang = [item.encode('utf-8') for item in lang]

    plot = movie.get('plot')
    if not plot:
        plot = 'NA'
    else:
        plot = [item.encode('utf-8') for item in plot]

    cast = movie.get('cast')
    if not cast:
        cast = 'NA'
    else:
        cast = cast[:5]
        main_cast = [name['name'].encode('utf-8') for name in cast]

    data = (movie['title'], main_cast, director, genres, lang, plot, imdbURL)
    cw.writerow(data)
    cur.execute("INSERT INTO movie_data (title, main_cast, director, genres, lang, plot, imdbURL) VALUES (%s, %s, %s, %s, %s, %s, %s)", (movie['title'], main_cast, str(director), genres, lang, plot, imdbURL))
    conn.commit()
    conn.close()

f1.write(out.getvalue())
