import nose.tools as n
import numpy as np
import pandas as pd
import sys
sys.path.append('code')
from recommend import Recommender
from CourseraTokenizer import CourseraTokenizer


def test_not_empty_requirement1():
    recommender = Recommender()
    req = ''
    n.assert_equal(recommender.not_empty_requirement(req), False)


def test_not_empty_requirement2():
    recommender = Recommender()
    req = '      '
    n.assert_equal(recommender.not_empty_requirement(req), False)


def test_not_empty_requirement3():
    recommender = Recommender()
    req = 'A passion for making sense of lots of unstructured data'
    n.assert_equal(recommender.not_empty_requirement(req), True)


def test_not_empty_requirement4():
    recommender = Recommender()
    req = '    A passion for making sense of lots of unstructured data'
    n.assert_equal(recommender.not_empty_requirement(req), True)


def test_not_empty_requirement5():
    recommender = Recommender()
    req = '    A passion for making sense of lots of unstructured data    '
    n.assert_equal(recommender.not_empty_requirement(req), True)


def test_get_top_courses():
    recommender = Recommender()
    lst = [0.30, 0.23, 0.34, 0.22]
    number = 2
    courses = ['c1', 'c2', 'c3', 'c4']
    course_names = ['course1', 'course2', 'course3', 'course4']
    n.assert_equal(recommender.get_top_courses(lst, number, courses, course_names),
                [('c3', 'course3', 0.34), ('c1', 'course1', 0.30)])


def test_get_missing_requirements():
    recommender = Recommender()
    lst = [0.2, 0.13, 0.04, 0.05, 0.049]
    preprocessed_requirements = ['r1', 'r2', 'r3', 'r4', 'r5']
    requirements = ['requirement1', 'requirement2', 'requirement3', 'requirement4', 'requirement5']
    n.assert_equal(recommender.get_missing_requirements(lst, preprocessed_requirements, requirements),
                [('r3', 'requirement3'), ('r5', 'requirement5')])


def test_remove_stopwords():
    recommender = Recommender()
    sentence = 'A passion for making sense of lots of unstructured data'
    n.assert_equal(recommender.remove_stopwords(sentence), 'A for making sense of lots of unstructured data')


def test_get_course_shortnames():
    ct = CourseraTokenizer()
    ct.set_df('data/courses_desc.json')
    n.assert_equal(ct.get_course_shortnames()[0], 'perceptivehunting')


def test_set_df():
    ct = CourseraTokenizer()
    ct.set_df('data/courses_desc.json')
    n.assert_equal(len(ct.get_course_shortnames()), 911)
