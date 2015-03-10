import pandas as pd
from bs4 import BeautifulSoup
import requests
import pymongo
from pymongo import MongoClient
import json


def get_coursera_course_data():
    '''
    Using the API from https://tech.coursera.org/app-platform/catalog/
    to build the cousera collection in mongodb which contain all the
    course infomation offered on coursera
    courses.json, collection coursedump.coursera are built
    '''

    client = MongoClient('mongodb://localhost:27017/')
    db = client.coursedump
    collection = db.coursera_desc
    url = 'https://api.coursera.org/api/catalog.v1/courses?fields=language,\
    shortDescription'
    response = requests.get(url)
    with open('courses_desc.json', 'w') as outfile:
        json.dump(response.json()['elements'], outfile)
    if response.status_code == 200:
        for course in response.json()['elements']:
            try:
                collection.insert(course)
            except pymongo.errors.DuplicateKeyError:
                pass
if __name__ == "__main__":
    get_coursera_course_data()
