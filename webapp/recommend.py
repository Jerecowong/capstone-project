import cPickle as pkl
import numpy as np
import nltk
import re
from sklearn.metrics.pairwise import linear_kernel
from CourseraTokenizer import CourseraTokenizer
from textblob import TextBlob


class Recommender(object):
    def __init__(self, ngram_range=(1, 1), use_stem=False, use_tagger=False):
        '''
        INPUT:
        - ngram_range: the lower and upper boundary of the range for different n-grams to be extracted
        - max_features: the vocabulary that only consider the top max_features ordered by term frequency
        - vectorizer: a collection of raw documents to a matrix of TF-IDF features
        - vectors: learn vocabulary and idf, return term-document matrix for Coursera courses.
        - use_tagger: a flag to extract noun phrases from requirements
        - coursera_courses: Coursera course short names
        - coursera_course_names Coursera course full names
        OUTPUT: None
        '''
        self.resume = None
        self.requirements = None
        self.preprocessed_requirements = None
        self.coursera_vectorizer = None
        self.coursera_vectors = None
        self.missing_requirements = None
        self.recommendations = []
        self.resume_vector = None
        self.requirement_vectors = None
        self.coursera_courses = None
        self.coursera_course_names = None
        self.ngram_range = ngram_range
        self.use_tagger = use_tagger

    def not_empty_requirement(self, requirement):
        '''
        Check if it is an empty line
        INPUT: STRING
        OUTPUT: BOOLEAN
        '''
        return False if re.match(r'^\s*$', requirement) else True

    def initialize_attributes(self, resume, requirements, coursera_vectorizer=None, coursera_vectors=None):
        self.resume = [resume]
        self.requirements = [requirement.strip() for requirement in requirements.split('\n') \
                            if self.not_empty_requirement(requirement)]
        # print self.requirements
        # print self.use_tagger
        self.preprocessed_requirements = self.requirements
        if self.use_tagger:
            self.preprocessed_requirements = [self.extract_noun_phrases_with_TextBlob(x) for x in self.requirements]
        # print self.requirements
        coursera_tokenizer = CourseraTokenizer(ngram_range=self.ngram_range)
        coursera_tokenizer.set_df('../data/courses_desc.json')
        coursera_tokenizer.set_vectors()
        self.coursera_vectorizer = coursera_tokenizer.get_vectorizer()
        self.coursera_vectors = coursera_tokenizer.get_vectors()
        self.coursera_courses = coursera_tokenizer.get_course_shortnames()
        self.coursera_course_names = coursera_tokenizer.get_course_names()

    def get_top_courses(self, lst, n, courses, course_names):
        '''
        Given a list of cosine similarities, find the indices with the highest n values.
        Return the short name and full name pair of courses for each of these indices.
        INPUT: LIST OF TURPLES, INTEGER, LIST
        OUTPUT: LIST

        Given a list of cosine similarities, find the indices with the highest n values.
        Return the courses for each of these indices.
        '''
        return [(courses[i], course_names[i], lst[i])for i in np.argsort(lst)[-1:-n - 1:-1]]

    def get_bottom_requirements(self, lst, n, preprocessed_requirements, requirements):
        '''
        Given a list of cosine similarities, find the indices with the lowest n values.
        Return the requirement and extracted requirment pair for each of these indices.
        INPUT: LIST, INTEGER, LIST OF TURPLES
        OUTPUT: LIST
        '''
        return [(preprocessed_requirements[i], requirements[i]) for i in np.argsort(lst)[0:n]]

    def get_missing_requirements(self, lst, preprocessed_requirements, requirements):
        '''
        Given a list of cosine similarities, find the indices with low value < 0.05.
        Return the requirement and extracted requirment pair for each of these indices.
        INPUT: LIST, INTEGER, LIST OF TURPLES
        OUTPUT: LIST
        '''
        return [(preprocessed_requirements[i], requirements[i]) for i in xrange(len(lst)) if lst[i] < 0.05]

    def remove_stopwords(self, sentence):
        '''
        Only keep the real skills in the requirement
        INPUT: STRING
        OUTPUT: STRING
        '''
        stopwords = ['experience', 'training', 'passion', 'background', 'skill', 'ability', 'skills', 'things',
                    'concepts', 'concept', 'traveling']
        return ' '.join([word for word in sentence.split() if word not in stopwords])

    def extract_noun_phrases_with_TextBlob(self, sentence):
        '''
        Only keep nouns for each line using TextBlob package
        INPUT: STRING
        OUTPUT: STRING
        '''
        sentence = self.remove_stopwords(sentence)
        text = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
        blob = TextBlob(text)
        return ' '.join(blob.noun_phrases)

    def extract_nouns(self, sentence):
        '''
        Only keep nouns for each line using nltk
        INPUT: STRING
        OUTPUT: STRING
        '''
        sentence = self.remove_stopwords(sentence)
        text = nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+', ' ', sentence))
        word_tags = nltk.pos_tag(text)
        return ' '.join([word_tag[0] for word_tag in word_tags if word_tag[1][:2] == 'NN'])

    def extract_nouns_verbings(self, sentence):
        '''
        Only keep nouns and verbs for each line using nltk
        INPUT: STRING
        OUTPUT: STRING
        '''
        sentence = self.remove_stopwords(sentence)
        text = nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+', ' ', sentence))
        word_tags = nltk.pos_tag(text)
        return ' '.join([word_tag[0] for word_tag in word_tags if (word_tag[1][:2] == 'NN' or word_tag == 'VBG')])

    def vectorize_resume(self):
        self.resume_vector = self.coursera_vectorizer.transform(self.resume)

    def vectorize_requirements(self):
        self.requirement_vectors = self.coursera_vectorizer.transform(self.requirements)

    def find_missing_skills(self):
        cosine_similarities = linear_kernel(self.requirement_vectors, self.resume_vector)
        # self.missing_requirements = self.get_bottom_requirements(cosine_similarities.flatten(),2,
        # self.preprocessed_requirements, self.requirements)
        self.missing_requirements = self.get_missing_requirements(cosine_similarities.flatten(),
             self.preprocessed_requirements, self.requirements)
        return self.missing_requirements

    def recommend(self):
        '''
        INPUT: None
        OUTPUT: names of the top 3 most relevant course recommendations
        '''
        missing_requirements_vectors = self.coursera_vectorizer.transform([item[0]
            for item in self.missing_requirements])
        cosine_similarities = linear_kernel(missing_requirements_vectors, self.coursera_vectors)
        for i, requirement in enumerate(self.missing_requirements):
            self.recommendations.append(self.get_top_courses(cosine_similarities[i], 3, self.coursera_courses,
                self.coursera_course_names))
        return self.recommendations
