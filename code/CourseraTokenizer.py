from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import cPickle as pkl


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
    with open('../data/coursera_tokenizer.pkl', 'wb') as handle:
        pkl.dump(coursera_tokenizer, handle)
    with open('../data/coursera_vectors.pkl', 'wb') as handle:
        pkl.dump(coursera_vectors, handle)
