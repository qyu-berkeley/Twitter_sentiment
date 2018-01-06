from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import pyhs2

#Create a engine for connecting to SQLite3.
#Assuming salaries.db is in your app root folder
app = Flask(__name__)
api = Api(app)

class Movie_Meta(Resource):
    def get(self):
        lines = [line.rstrip('\n') for line in open('movienames.txt')]
        return {'moviename': [lines]}

api.add_resource(Movie_Meta, '/movies')

if __name__ == '__main__':
            #print "Hi3"
    app.run()
