from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import cPickle as pkl


def get_top_courses(lst, n, courses):
    '''
    to build the cousera collection in mongodb which contain all the
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of cosine similarities, find the indices with the highest n values.
    Return the courses for each of these indices.
    '''
    return [(courses[i], lst[i])for i in np.argsort(lst)[-1:-n - 1:-1]]


class CourseraTokenizer(object):
    def __init__(self, use_stem=False):
        '''
        INPUT:
        - ngram_range:
        - max_features:
        - vectorizer:
        - vectors:
        - df:
        OUTPUT: None
        '''
        self.ngram_range = (1, 1)
        self.max_features = 1000
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=self.ngram_range,
                                    max_features=self.max_features)
        self.vectors = None
        self.df = None

    def get_df(self):
        return self.df

    def set_df(self, file_name):
        self.df = pd.read_json(file_name)
        self.df['description'] = self.df['name'] + ' ' + self.df['shortDescription']

    def get_vectors(self):
        return self.vectors

    def get_descriptions(self):
        '''
        Maybe I need use list(df['description'].values)
        '''
        return self.df['description']

    def set_vectors(self):
        docs = self.get_descriptions()
        self.vectors = self.vectorizer.fit_transform(docs).toarray()

    def get_vectorizer(self):
        return self.vectorizer

    def get_course_shortnames(self):
        return self.df['shortName']

if __name__ == '__main__':
    coursera_tokenizer = CourseraTokenizer()
    coursera_tokenizer.set_df('../data/courses_desc.json')
    coursera_tokenizer.set_vectors()
    coursera_vectorizer = coursera_tokenizer.get_vectorizer()
    coursera_vectors = coursera_tokenizer.get_vectors()
    coursera_courses = coursera_tokenizer.get_course_shortnames()
    requirements = ["A passion for making sense of lots of unstructured data",
                    "Experience with NLP, classification, graph mining, and/or recommender systems",
                    "Proficient with SQL and experience with one or more DB query clients (e.g. MySQL)",
                    "Solid experience in Java and a scripting language (Python, Perl,etc.)",
                    "Bonus: Experience with R, D3, Map/Reduce, HIVE/Pig, MongoDB",
                    "Bonus: Experience with applying solutions to languages such as Japanese, German, French",
                    "MS or PhD in a technical degree",
                    "Enthusiasm for working in a fun, dynamic startup environment",
                    "You like to shop online and you don't mind getting reimbursed to buy more stuff!",
                    "You can code in the presence of flying ping pong balls."]
    requirements_vectors = coursera_vectorizer.transform(requirements)
    cosine_similarities = linear_kernel(requirements_vectors, coursera_vectors)
    for i, requirement in enumerate(requirements):
        print requirement
        print get_top_courses(cosine_similarities[i], 3, coursera_courses)
        print
    with open('../data/coursera_tokenizer.pkl', 'wb') as handle:
        pkl.dump(coursera_tokenizer, handle)
    with open('../data/coursera_vectors.pkl', 'wb') as handle:
        pkl.dump(coursera_vectors, handle)
