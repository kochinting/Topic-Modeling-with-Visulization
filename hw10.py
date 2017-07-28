""""
605.433 Social Media Analysis
Module 10 Assignment
Chin-Ting Ko

This program is to perform topic modeling for social media content with gensim and visualize with PyLDAvis.
reference: lecture content: RedditTopicModeler.py
"""

__author__ = 'Ko, Chin-Ting'

import praw
import json
import gensim

# setup for tokenization and stopwords
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from gensim import corpora, models, similarities
#from collections import defaultdict

stop_words = set(stopwords.words('english'))
stop_words.update(['.',  ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

file = open("redditprocess.txt", "w")
r = praw.Reddit(user_agent='Myi2XhjayG3r39m8ClEPD30xCh0')
for post in r.get_subreddit('music') \
             .get_comments(limit=500):
    file.write(post.body.encode('utf-8').strip() + '\n')
    words = [i.lower() for i in wordpunct_tokenize(post.body) if i.lower() not in stop_words]
    #print (words)
    all_tokens = ' '.join(words)
#print (words)
#dictionary = corpora.Dictionary(words)
#print (dictionary)

# Remove words that appear only once

texts = words
# remove words that appear only once
#all_tokens = sum(texts, [])
tokens_once = set(words for words in set(all_tokens) if all_tokens.count(words) == 1)
#print(tokens_once)
texts = [[words for words in texts if words not in tokens_once]
         for words in all_tokens]
#print (texts)
    #file.write(text + '\n')
#    file.write(post.body + '\n')
# Setup for Document Matrix
#Setup gensim dictionary

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('reddit.mm', corpus)
#print(corpus)

lda = gensim.models.LdaModel(corpus, id2word=dictionary, alpha='auto', num_topics=10)
for i in lda.show_topics():
    print (i)
#convert ot BOW vectors

import pyLDAvis.gensim

topic_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(topic_vis)

file.close