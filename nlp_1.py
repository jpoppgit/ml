#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Practical example of NLP.

    Inspired by
    https://python-course.eu/machine-learning/natural-language-processing-with-python.php

"""

from sklearn.feature_extraction import text
import pandas as pd

import inspect
import logging

logging.basicConfig(level=logging.INFO)

corpus = ["To be, or not to be, that is the question:",
          "Whether 'tis nobler in the mind to suffer",
          "The slings and arrows of outrageous fortune,"]

def bag_of_words():
    sCurrentFunction = inspect.stack()[0][3]
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


# Launch different concepts.
bag_of_words()
