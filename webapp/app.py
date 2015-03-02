from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import cPickle as pkl
from recommend import Recommender
#from parser import TextParser


app = Flask(__name__)

@app.route('/')
def submit_forms():
    return render_template('query.html')
    
@app.route('/recommend', methods=['POST'])
def recommend():
    resume = request.form.get('resume', None) 
    requirements = request.form.get('requirements', None)
    #course_recommedations = [u'datascitoolbox', u'pythonlearn', u'frenchrev', u'programming2', u'massiveteaching']
    '''
    with open('data/coursera_vectorizer.pkl', 'rb') as handle:
        coursera_vectorizer = pkl.load(handle)
    with open('data/coursera_vectors.pkl', 'rb') as handle:
        coursera_vectors = pkl.load(handle)
    '''
    recommender = Recommender()
    #recommender.initialize_attributes(resume, requirements, coursera_vectorizer, coursera_vectors)
    recommender.initialize_attributes(resume, requirements)
    recommender.vectorize_resume()
    recommender.vectorize_requirements()
    recommender.find_missing_skills()
    course_recommedations = recommender.recommend()
    return render_template('recommend.html', data=course_recommedations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)



