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
    '''
    with open('data/coursera_vectorizer.pkl', 'rb') as handle:
        coursera_vectorizer = pkl.load(handle)
    with open('data/coursera_vectors.pkl', 'rb') as handle:
        coursera_vectors = pkl.load(handle)
    '''
    recommender = Recommender(ngram_range=(1, 1), use_tagger=True)
    #recommender.initialize_attributes(resume, requirements, coursera_vectorizer, coursera_vectors)
    recommender.initialize_attributes(resume, requirements)
    recommender.vectorize_resume()
    recommender.vectorize_requirements()
    missing_requirement_pairs = recommender.find_missing_skills()
    missing_requirements = [item[1] for item in missing_requirement_pairs]
    course_recommedations = recommender.recommend()
    '''
    print "in App"
    print "missing_requirements"
    print missing_requirements
    print len(missing_requirements)
    '''
    if len(missing_requirements) > 0:
        return render_template('recommend.html', data=zip(missing_requirements, course_recommedations))
    return "You Meet all the requirements"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)



