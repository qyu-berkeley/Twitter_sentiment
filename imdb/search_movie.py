from imdb import IMDb
import StringIO
import csv
import sys

out = StringIO.StringIO()
f1 = open('title_search.csv', 'w')
ia = IMDb('http')

search_title = sys.argv[1]
cw = csv.writer(out)
header = ("Title", "Cast", "Director", "Genre", "language", "Plot_summary", "imdb_URL")
cw.writerow(header)

movies = ia.search_movie(search_title)

for movie_obj in movies:

    # Get movie by movieID
    id = movie_obj.movieID
    movie = ia.get_movie(id)

    imdbURL = ia.get_imdbURL(movie)
    if not imdbURL:
        imdbURL = 'NA'

    genres = movie.get('genres')
    if not genres:
        genres = 'NA'

    director = movie.get('director')
    if not director:
        director = 'NA'
    else:
        director = director[0]

    lang = movie.get('lang')
    if not lang:
        lang = 'NA'

    plot = movie.get('plot')
    if not plot:
        plot = 'NA'

    cast = movie.get('cast')
    if not cast:
        cast = 'NA'
    else:
        cast = cast[:5]
        main_cast = [name['name'] for name in cast]

    data = (movie['title'], main_cast, director, genres, lang, plot, imdbURL)
    cw.writerow(data)
    print data

print out.getvalue()
f1.write(out.getvalue())
