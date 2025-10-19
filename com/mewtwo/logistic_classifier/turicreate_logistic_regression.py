"""Logistic Regression using Turi Create"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

movies = tc.SFrame('/home/mewtwo/py_workspace/grokking_self_learning/com/mewtwo/logistic_classifier/IMDB_Dataset.csv')
movies['words'] = tc.text_analytics.count_words(movies['review'])
model = tc.logistic_classifier.create(movies, features='words', target='sentiment')
movies