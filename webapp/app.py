from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle
#from parser import TextParser


app = Flask(__name__)

@app.route('/')
def submit_forms():
    return render_template('query.html')
    
@app.route('/recommend', methods=['POST'])
def recommend():
	resume = request.form.get('resume', None) 
	requirements = request.form.get('requirements', None)
	#shortnames = [u'datascitoolbox', u'pythonlearn', u'frenchrev', u'programming2', u'massiveteaching']
	recommender = Recommnder()
	recommender.set_resume(resume)
	recommender.set_requirements(requirements)
	course_recommedations = recommender.get_recommedations()
	return render_template('recommend.html', data=course_recommedations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)



