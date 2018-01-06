from imdb import IMDb

i = IMDb('sql', uri='postgresql://localhost/imdb')

resList = i.search_movie('Moana')

for x in resList: print x
ti = resList[0]
i.update(ti)
print ti['director'][0]
