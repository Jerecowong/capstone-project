from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import cPickle as pkl


class CourseraTokenizer(object):
    def __init__(self, ngram_range=(1, 1), use_stem=False):
        '''
        INPUT:
        - ngram_range: lower and upper boundary of the range for n-grams
        - max_features: vocabulary of the top max_features by term frequency
        - vectorizer: collection of raw documents to a matrix of TF-IDF
        - vectors: Learn vocabulary and idf, return term-document matrix.
        - df: pandas.DataFrame for Coursera courses
        OUTPUT: None
        '''
        self.ngram_range = ngram_range
        self.max_features = None
        self.vectorizer = TfidfVectorizer(stop_words='english',
            ngram_range=self.ngram_range, max_features=self.max_features)
        self.vectors = None
        self.df = None

    def get_df(self):
        return self.df

    def set_df(self, file_name):
        self.df = pd.read_json(file_name)
        self.df['description'] = self.df['name'] + ' ' \
            + self.df['shortDescription']

    def get_vectors(self):
        return self.vectors

    def get_descriptions(self):
        return self.df['description']

    def set_vectors(self):
        docs = self.get_descriptions()
        self.vectors = self.vectorizer.fit_transform(docs).toarray()

    def get_vectorizer(self):
        return self.vectorizer

    def get_course_shortnames(self):
        return self.df['shortName']

    def get_course_names(self):
        return self.df['name']

if __name__ == '__main__':
    coursera_tokenizer = CourseraTokenizer()
    coursera_tokenizer.set_df('../data/courses_desc.json')
    coursera_tokenizer.set_vectors()
    coursera_vectorizer = coursera_tokenizer.get_vectorizer()
    coursera_vectors = coursera_tokenizer.get_vectors()
    coursera_courses = coursera_tokenizer.get_course_shortnames()
    # Save the tokenizer and vectors
    with open('../data/coursera_tokenizer.pkl', 'wb') as handle:
        pkl.dump(coursera_tokenizer, handle)
    with open('../data/coursera_vectors.pkl', 'wb') as handle:
        pkl.dump(coursera_vectors, handle)
