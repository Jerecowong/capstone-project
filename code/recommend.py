import cPickle as pkl
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from CourseraTokenizer import CourseraTokenizer
from textblob import TextBlob


class Recommender(object):
    def __init__(self, ngram_range=(1, 1), use_stem=False, use_tagger=False):
        '''
        INPUT:
        - ngram_range: lower and upper boundary of the range for n-grams
        - max_features: vocabulary of the top max_features by term frequency
        - vectorizer: collection of raw documents to a matrix of TF-IDF
        - vectors: Learn vocabulary and idf, return term-document matrix.
        - use_tagger: flag to extract noun phrases from requirements
        - coursera_courses: Coursera course short names
        - coursera_course_names Coursera course full names
        OUTPUT: None
        '''
        self.resume = None
        self.requirements = None
        self.preprocessed_requirements = None
        self.resume_vectorizer = None
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
        self.use_stem = use_stem

    def fit(self, resume, requirements):
        '''
        initialize the Recommender
        '''
        self.initialize_attributes(resume, requirements)
        self.vectorize_resume()
        self.vectorize_requirements()

    def not_empty_requirement(self, requirement):
        '''
        Check if it is an empty line
        INPUT: STRING
        OUTPUT: BOOLEAN
        '''
        return False if re.match(r'^\s*$', requirement) else True

    def initialize_attributes(self, resume, requirements,
            coursera_vectorizer=None, coursera_vectors=None):
        self.resume = [resume]
        self.requirements = [requirement.strip() for requirement in
            requirements.split('\n') if self.not_empty_requirement(requirement)]
        self.preprocessed_requirements = self.requirements
        if self.use_tagger:
            self.preprocessed_requirements = [self.extract_nouns_TextBlob(x)
                for x in self.requirements]
        coursera_tokenizer = CourseraTokenizer(ngram_range=self.ngram_range,
                            use_stem=self.use_stem)
        coursera_tokenizer.set_df('data/courses_desc.json')
        coursera_tokenizer.set_vectors()
        self.coursera_vectorizer = coursera_tokenizer.get_vectorizer()
        self.coursera_vectors = coursera_tokenizer.get_vectors()
        self.coursera_courses = coursera_tokenizer.get_course_shortnames()
        self.coursera_course_names = coursera_tokenizer.get_course_names()
        self.resume_vectorizer = TfidfVectorizer(stop_words='english',
            ngram_range=(1, 1))

    def get_top_courses(self, lst, n, courses, course_names):
        '''
        Given cosine similarity list , find indices with the highest n values.
        Return short and full name pair of courses for each of the indices.
        INPUT: LIST OF TURPLES, INTEGER, LIST
        OUTPUT: LIST
        '''
        return [(courses[i], course_names[i], lst[i])
                for i in np.argsort(lst)[-1:-n - 1:-1]]

    def stematize_descriptions(self, descriptions):
        snowball = SnowballStemmer('english')
        stematize = lambda desc: ' '.join(snowball.stem(word)
                for word in desc.split())
        return [stematize(re.sub(r'[^\x00-\x7F]+', ' ', desc))
                for desc in descriptions]

    def filter_courses(self, courses_triple, threshold=0.10):
        '''
        Keep the courses with similarity score higher than threshold
        '''
        return [item for item in courses_triple if item[2] >= threshold]

    def get_bottom_requirements(self, lst, n, preprocessed_requirements,
                             requirements):
        '''
        Given cosine similarity list, find indices with the lowest n values.
        Return original and extracted requirment pair for each of the indices.
        INPUT: LIST, INTEGER, LIST OF TURPLES
        OUTPUT: LIST
        '''
        return [(preprocessed_requirements[i], requirements[i])
                for i in np.argsort(lst)[0:n]]

    def get_missing_requirements(self, lst, preprocessed_requirements,
                                requirements):
        '''
        Given a list of cosine similarities, find the indices with value < 0.05
        Return original and extracted requirment pair for each of these indices.
        INPUT: LIST, INTEGER, LIST OF TURPLES
        OUTPUT: LIST
        '''
        return [(preprocessed_requirements[i], requirements[i])
                for i in xrange(len(lst)) if lst[i] < 0.05]

    def remove_stopwords(self, sentence):
        '''
        Only keep the real skills in the requirement
        INPUT: STRING
        OUTPUT: STRING
        '''
        stopwords = ['experience', 'training', 'passion', 'background', 'skill',
            'ability', 'skills', 'things', 'concepts', 'concept', 'traveling']
        sentence = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
        return ' '.join([word for word in sentence.split()
                if word.lower() not in stopwords])

    def extract_nouns_TextBlob(self, sentence):
        '''
        Only keep nouns for each line using TextBlob package
        INPUT: STRING
        OUTPUT: STRING
        '''
        text = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
        sentence = self.remove_stopwords(sentence)
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
        return ' '.join([word_tag[0] for word_tag in word_tags
                if word_tag[1][:2] == 'NN'])

    def extract_nouns_verbs(self, sentence):
        '''
        Only keep nouns and verbs for each line using nltk
        INPUT: STRING
        OUTPUT: STRING
        '''
        sentence = self.remove_stopwords(sentence)
        text = nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+', ' ', sentence))
        word_tags = nltk.pos_tag(text)
        return ' '.join([word_tag[0] for word_tag in word_tags
            if (word_tag[1][:2] == 'NN' or word_tag[1][:2] == 'VB')])

    def vectorize_resume(self):
        resume = self.resume
        if self.use_stem:
            resume = self.stematize_descriptions(resume)
        self.resume_vector = self.resume_vectorizer.fit_transform(resume)

    def vectorize_requirements(self):
        requirements = self.requirements
        if self.use_stem:
            requirements = self.stematize_descriptions(requirements)
        self.requirement_vectors = self.resume_vectorizer.transform(
            requirements)

    def find_missing_skills(self):
        '''
        INPUT: None
        OUTPUT: List of pairs of missing requirement in original and
        extracted form
        '''
        cosine_similarities = linear_kernel(self.requirement_vectors,
                self.resume_vector)
        self.missing_requirements = self.get_missing_requirements(
            cosine_similarities.flatten(),
            self.preprocessed_requirements,
            self.requirements)
        return self.missing_requirements

    def recommend(self):
        '''
        INPUT: None
        OUTPUT: names of the top 3 most relevant course recommendations
        '''
        missing_requirements = [item[0] for item in self.missing_requirements]
        if self.use_stem:
            missing_requirements = self.stematize_descriptions(
                missing_requirements)
        missing_requirements_vectors = self.coursera_vectorizer.transform(
            missing_requirements)
        cosine_similarities = linear_kernel(missing_requirements_vectors,
            self.coursera_vectors)
        for i, requirement in enumerate(self.missing_requirements):
            self.recommendations.append(self.filter_courses(
                self.get_top_courses(cosine_similarities[i], 3,
                    self.coursera_courses, self.coursera_course_names)))
        return self.recommendations
