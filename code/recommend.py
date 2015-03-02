import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from CourseraTokenizer import CourseraTokenizer

def get_top_courses(lst, n, courses):
    '''
    to build the cousera collection in mongodb which contain all the
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of cosine similarities, find the indices with the highest n values.
    Return the courses for each of these indices.
    '''
    return [(courses[i], lst[i])for i in np.argsort(lst)[-1:-n - 1:-1]]

def get_bottom_requirements(lst, n, requirements):
    '''
    to build the cousera collection in mongodb which contain all the
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of cosine similarities, find the indices with the highest n values.
    Return the courses for each of these indices.
    '''
    return [requirements[i] for i in np.argsort(lst)[0:n]]

class Recommender(object):
    '''
    Recommender class 
    '''

    def __init__(self):
        '''
        pass
        '''
        self.resume = None
        self.requirements = None
        self.coursera_vectorizer = None
        self.coursera_vectors = None
        self.missing_requirements = None
        self.recommendations = []
        self.resume_vector = None
        self.requirement_vectors = None
        self.coursera_courses = None

    def initialize_attributes(self, resume, requirements, coursera_vectorizer=None, coursera_vectors=None):
        self.resume = [resume]
        self.requirements = [requirement for requirement in requirements.split('\n')]
        coursera_tokenizer = CourseraTokenizer()
        coursera_tokenizer.set_df('../data/courses_desc.json')
        coursera_tokenizer.set_vectors()
        self.coursera_vectorizer = coursera_tokenizer.get_vectorizer()
        self.coursera_vectors = coursera_tokenizer.get_vectors()
        self.coursera_courses = coursera_tokenizer.get_course_shortnames()
        #self.coursera_vectorizer = coursera_vectorizer
        #self.coursera_vectors = coursera_vectors

    def vectorize_resume(self):
        self.resume_vector = self.coursera_vectorizer.transform(self.resume)

    def vectorize_requirements(self):
        self.requirement_vectors = self.coursera_vectorizer.transform(self.requirements)

    def find_missing_skills(self):
        cosine_similarities = linear_kernel(self.requirement_vectors, self.resume_vector)
        self.missing_requirements = get_bottom_requirements(cosine_similarities, 2, self.requirements)

    def recommend(self):
        '''
        INPUT: np array of distances
        OUTPUT: names of recommendations
        '''
        missing_requirements_vectors = self.coursera_vectorizer.transform(self.missing_requirements)
        cosine_similarities = linear_kernel(missing_requirements_vectors, self.coursera_vectors)
        for i, requirement in enumerate(self.missing_requirements):
            self.recommendations.append(get_top_courses(cosine_similarities[i], 3, self.coursera_courses))
        return self.recommendations

