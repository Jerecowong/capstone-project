import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

class Recommender(object):
    '''
    
    Recommender class 
    
    '''

    def __init__(self):
        '''
        pass
        '''
        pass
        
    def vectorize_resume(self):
        '''
        pass
        '''
        pass


    def vectorize_requirements(self):
    	pass

    def find_missing_skills(self):
    	pass

    def recommend(self):
        '''
        INPUT: np array of distances
        OUTPUT: names of recommendations
        '''
        pass