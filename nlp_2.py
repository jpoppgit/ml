#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Practical example of NLP.

    Inspired by
    https://python-course.eu/machine-learning/natural-language-processing-classification.php

    Resource
    https://python-course.eu/books/the_voyage_out_virginia_woolf.txt

    Language predictions
        wget -O it_alessandro_manzoni_i_promessi_sposi.txt https://www.gutenberg.org/cache/epub/45334/pg45334.txt
        wget -O es_antonio_de_alarcon_novelas_cortas.txt https://www.gutenberg.org/cache/epub/15532/pg15532.txt
        wget -O de_nietzsche_also_sprach_zarathustra.txt https://www.gutenberg.org/cache/epub/7205/pg7205.txt
        wget -O nl_lodewijk_van_deyssel.txt https://www.gutenberg.org/cache/epub/10820/pg10820.txt
        wget -O de_goethe_leiden_des_jungen_werther2.txt https://www.gutenberg.org/cache/epub/2408/pg2408.txt
        wget -O se_august_strindberg_röda_rummet.txt https://www.gutenberg.org/cache/epub/57052/pg57052.txt
        wget -O it_amato_gennaro_una_sfida_al_polo.txt https://www.gutenberg.org/files/58415/58415-0.txt
        wget -O nl_cornelis_johannes_kieviet_Dik_Trom_en_sijn_dorpgenooten.txt https://www.gutenberg.org/cache/epub/10971/pg10971.txt
        wget -O fr_emile_zola_la_bete_humaine.txt https://www.gutenberg.org/cache/epub/16852/pg16852.txt
        wget -O se_selma_lagerlöf_bannlyst.txt https://www.gutenberg.org/cache/epub/39147/pg39147.txt
        wget -O de_goethe_leiden_des_jungen_werther1.txt https://www.gutenberg.org/cache/epub/2407/pg2407.txt
        wget -O en_virginia_woolf_night_and_day.txt https://www.gutenberg.org/files/1245/1245-0.txt
        wget -O es_mguel_de_cervantes_don_cuijote.txt https://www.gutenberg.org/files/2000/2000-0.txt
        wget -O en_herman_melville_moby_dick.txt https://www.gutenberg.org/files/15/15-0.txt
        wget -O dk_andreas_lauritz_clemmensen_beskrivelser_og_tegninger.txt https://www.gutenberg.org/files/61722/61722-0.txt
        wget -O fr_emile_zola_germinal.txt https://www.gutenberg.org/cache/epub/5711/pg5711.txt
"""
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pandas as pd

import random
import logging

logging.basicConfig(level=logging.INFO)

def text2paragraphs(filename, min_size=1):
    """ A text contained in the file 'filename' will be read 
    and chopped into paragraphs.
    Paragraphs with a string length less than min_size will be ignored.
    A list of paragraph strings will be returned"""
    
    txt = open(filename).read()
    paragraphs = [para for para in txt.split("\n\n") if len(para) > min_size]
    return paragraphs

labels = ['Virginia Woolf', 'Samuel Butler', 'Herman Melville', 
          'David Herbert Lawrence', 'Daniel Defoe', 'James Joyce']

files = ['night_and_day_virginia_woolf.txt', 'the_way_of_all_flash_butler.txt',
         'moby_dick_melville.txt', 'sons_and_lovers_lawrence.txt',
         'robinson_crusoe_defoe.txt', 'james_joyce_ulysses.txt']

path = "./books/"

data = []
targets = []
counter = 0
for fname in files:
    paras = text2paragraphs(path + fname, min_size=150)
    data.extend(paras)
    targets += [counter] * len(paras)
    counter += 1

data_targets = list(zip(data, targets))
# create random permuation on list:
data_targets = random.sample(data_targets, len(data_targets))

data, targets = list(zip(*data_targets))

res = train_test_split(data, targets, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_targets, test_targets = res 
logging.info(str(len(train_data))+' '+str(len(test_data))+' '+str(len(train_targets))+' '+str(len(test_targets)))

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)

train_vectors = vectorizer.fit_transform(train_data)

# creating a classifier
classifier = MultinomialNB(alpha=.01)
classifier.fit(train_vectors, train_targets)

vectors_test = vectorizer.transform(test_data)

predictions = classifier.predict(vectors_test)
accuracy_score = metrics.accuracy_score(test_targets, 
                                        predictions)
f1_score = metrics.f1_score(test_targets, 
                            predictions, 
                            average='macro')

print("accuracy score: ", accuracy_score)
print("F1-score: ", f1_score)

# Test this classifier now with a different book of Virginia Woolf.
paras = text2paragraphs(path + "the_voyage_out_virginia_woolf.txt", min_size=250)

first_para, last_para = 100, 500
vectors_test = vectorizer.transform(paras[first_para: last_para])
#vectors_test = vectorizer.transform(["To be or not to be"])

predictions = classifier.predict(vectors_test)
print(predictions)
targets = [0] * (last_para - first_para)
accuracy_score = metrics.accuracy_score(targets, 
                                        predictions)
precision_score = metrics.precision_score(targets, 
                                          predictions, 
                                          average='macro')
f1_score = metrics.f1_score(targets, 
                            predictions, 
                            average='macro')

print("accuracy score: ", accuracy_score)
print("precision score: ", accuracy_score)
print("F1-score: ", f1_score)

# We want to train now a Neural Network.
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
print('\n\nTraining a neural network ...')
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
vectors = vectorizer.fit_transform(train_data)

print("Creating a classifier. This will take some time!")
classifier = MLPClassifier(random_state=1, max_iter=300).fit(vectors, train_targets)

vectors_test = vectorizer.transform(test_data)

predictions = classifier.predict(vectors_test)
accuracy_score = metrics.accuracy_score(test_targets, 
                                        predictions)
f1_score = metrics.f1_score(test_targets, 
                            predictions, 
                            average='macro')

print("accuracy score: ", accuracy_score)
print("F1-score: ", f1_score)


print('\n\n-----------------------------------')
print('Language prediction ...')
print(os.listdir("books/various_languages"))

labels = ['Virginia Woolf', 'Samuel Butler', 'Herman Melville', 
          'David Herbert Lawrence', 'Daniel Defoe', 'James Joyce']

path = "books/various_languages/"

files = os.listdir(path)
labels = {fname[:2] for fname in files if fname.endswith(".txt")}
labels = sorted(list(labels))
print(labels)

data    = []
targets = []
for fname in files:
    if fname.endswith(".txt"):
        paras = text2paragraphs(path + fname, min_size=150)
        data.extend(paras)
        country = fname[:2]
        index = labels.index(country)
        targets += [index] * len(paras)
print(str(len(data))+' '+str(len(targets)))

data_targets = list(zip(data, targets))
# create random permuation on list:
data_targets = random.sample(data_targets, len(data_targets))
data, targets = list(zip(*data_targets))

res = train_test_split(data, targets, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_targets, test_targets = res

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)

vectors = vectorizer.fit_transform(train_data)

# creating a classifier
classifier = MultinomialNB(alpha=.01)
classifier.fit(vectors, train_targets)

vectors_test = vectorizer.transform(test_data)

predictions = classifier.predict(vectors_test)
accuracy_score = metrics.accuracy_score(test_targets, 
                                        predictions)
f1_score = metrics.f1_score(test_targets, 
                            predictions, 
                            average='macro')

print("accuracy score: ", accuracy_score)
print("F1-score: ", f1_score)

# Check with some abitrary text in different languages.
some_texts = ["Es ist nicht von Bedeutung, wie langsam du gehst, solange du nicht stehenbleibst.",
              "Man muss das Unmögliche versuchen, um das Mögliche zu erreichen.",
              "It's so much darker when a light goes out than it would have been if it had never shone.",
              "Rien n'est jamais fini, il suffit d'un peu de bonheur pour que tout recommence.",
              "Girano le stelle nella notte ed io ti penso forte forte e forte ti vorrei"]
vtest = vectorizer.transform(some_texts)
predictions = classifier.predict(vtest)
for label in predictions:
    print(label, labels[label])
