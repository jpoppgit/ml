#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Practical example of NLP.

    Inspired by
    https://python-course.eu/machine-learning/natural-language-processing-with-python.php

"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pandas as pd
import numpy as np

import inspect
import logging

logging.basicConfig(level=logging.INFO)

def bag_of_words():
    sCurrentFunction = inspect.stack()[0][3]

    corpus = ["To be, or not to be, that is the question:",
            "Whether 'tis nobler in the mind to suffer",
            "The slings and arrows of outrageous fortune,"]

    vectorizer = text.CountVectorizer()
    logging.info(sCurrentFunction+': corpus: '+str(corpus))
    #logging.info(sCurrentFunction+': '+str(vectorizer))
    vectorizer.fit(corpus)
    logging.info(sCurrentFunction+': vocabulary: '+str(vectorizer.vocabulary_))

    token_count_matrix = vectorizer.transform(corpus)
    logging.info(sCurrentFunction+': token count: '+str(token_count_matrix))

    dense_tcm = token_count_matrix.toarray()
    pd.DataFrame(data=dense_tcm, 
                index=['corpus_0', 'corpus_1', 'corpus_2'],
                columns=vectorizer.get_feature_names_out())

    word = "be"
    i = 1
    j = vectorizer.vocabulary_[word]
    logging.info("number of times '" + word + "' occurs in:")
    for i in range(len(corpus)):
        logging.info("    '" + corpus[i] + "': " + str(dense_tcm[i][j]))

def word_importance():
    sCurrentFunction = inspect.stack()[0][3]
    logging.info('\n\n')

    corpus = ["It does not matter what you are doing, just do it!",
            "Would you work if you won the lottery?",
            "You like Python, he likes Python, we like Python, everybody loves Python!"
            "You said: 'I wish I were a Python programmer'",
            "You can stay here, if you want to. I would, if I were you."
            ]

    vectorizer = text.CountVectorizer()
    vectorizer.fit(corpus)

    token_count_matrix = vectorizer.transform(corpus)
    logging.info(sCurrentFunction+': token_count_matrix: '+str(token_count_matrix))

    tf_idf = text.TfidfTransformer()
    tf_idf.fit(token_count_matrix)

    sTarget = 'python'
    logging.info(sCurrentFunction+': '+sTarget+': '+ str( tf_idf.idf_[vectorizer.vocabulary_[sTarget]] ))

    # check how often the word 'would' occurs in the the i'th sentence:
    #vectorizer.vocabulary_['would']
    sTarget = 'would'
    da = vectorizer.transform(corpus).toarray()
    i = 0
    word_ind = vectorizer.vocabulary_[sTarget]
    logging.info(sCurrentFunction+': '+sTarget+': '+str(da[i][word_ind]))
    logging.info(sCurrentFunction+': '+sTarget+': '+str(da[:,word_ind]))

def working_with_real_data():
    sCurrentFunction = inspect.stack()[0][3]
    logging.info('\n\n')

    # Create our vectorizer
    vectorizer = CountVectorizer()

    # Let's fetch all the possible text data
    newsgroups_data = fetch_20newsgroups()
    logging.info(sCurrentFunction+': '+newsgroups_data.data[0])

    vectorizer.fit(newsgroups_data.data)

    counter = 0
    n = 10
    for word, index in vectorizer.vocabulary_.items():
        logging.info(sCurrentFunction+': '+str(word)+' '+str(index))
        counter += 1
        if counter > n:
            break

def stop_words():
    sCurrentFunction = inspect.stack()[0][3]
    logging.info('\n\n')

    newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',
                                        remove=('headers', 'footers', 'quotes'))

    corpus = ["A horse, a horse, my kingdom for a horse!",
          "Horse sense is the thing a horse has which keeps it from betting on people."
          "I’ve often said there is nothing better for the inside of the man, than the outside of the horse.",
          "A man on a horse is spiritually, as well as physically, bigger then a man on foot.",
          "No heaven can heaven be, if my horse isn’t there to welcome me."]

    cv = CountVectorizer(input=corpus,
                        stop_words=["my", "for","the", "has", "than", "if", 
                                    "from", "on", "of", "it", "there", "ve",
                                    "as", "no", "be", "which", "isn", "to", 
                                    "me", "is", "can", "then"])
    count_vector = cv.fit_transform(corpus)
    count_vector.shape

    logging.info(sCurrentFunction+': '+str(cv.vocabulary_))

    vectorizer = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)

    vectors = vectorizer.fit_transform(newsgroups_train.data)

    # creating a classifier
    classifier = MultinomialNB(alpha=.01)
    classifier.fit(vectors, newsgroups_train.target)

    vectors_test = vectorizer.transform(newsgroups_test.data)

    predictions = classifier.predict(vectors_test)
    accuracy_score = metrics.accuracy_score(newsgroups_test.target, 
                                            predictions)
    f1_score = metrics.f1_score(newsgroups_test.target, 
                                predictions, 
                                average='macro')

    print("accuracy score: ", accuracy_score)
    print("F1-score: ", f1_score)

# Launch different concepts.
bag_of_words()
word_importance()
working_with_real_data()
stop_words()
