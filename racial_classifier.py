#################################
# Copyright 2018 Aaron Mauro
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################

print("Importing...")
import requests
from lxml import html
from string import punctuation

import random
import pygal
import pickle
import os
import sys
import csv
import time
import multiprocessing
from statistics import mode

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI

class VotingClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


"""
#Run this section once to download racist / national terms from Wikipedia

print("Finished imports.\nDownloading ethnic slurs from Wikipedia")

page = requests.get("https://en.wikipedia.org/wiki/List_of_ethnic_slurs")
tree = html.fromstring(page.content)

terms = tree.xpath('//dt/text()')

raw_terms = ["".join([ch for ch in word if ch not in punctuation]).lower() for word in terms]

stop_list = []

stop_terms = ['froggy','frog','gin','yam','crow','aunt','touch','moon','plural','or','are','us','uk','and','of','yellow','white','plural','or','are','us','uk','and','of','the','in','from','also','timber','face','lo','ball','grant','head','flip']

additional_terms = [bugger, Turk, Greek, coolie, blackamoor, ethiop, jew, tartar, bogtrotter,vandal, goth, macaroni, dago, hottentot, yankee, cracker, frog, kaffir, nigger, coon, Frenchy, wi-wi, sheeny greaser, gringo, canuck, sambo, Jap, yid, mick, limey, kike, hun, chink, wop, boche, fritz, jerry, kraut, pom, wog, spick, eyetie, ofay, spaghetti, wetback, nip, gook, anglo,slant, slope, munt, honkie, Paki, heathen, infidel, paynim, savage, alien, intruder, barbarian, foreigner, native,]

term_set = [word_tokenize(term) for term in raw_terms]
racist_terms = []
for t in term_set:
    for w in t:
        if w[0].isalpha():
            if w not in set(stop_list):
                racist_terms.append(w)

for term in additional_terms:
    racist_terms.append(term.lower())
    
print("We have",len(racist_terms),"racial slurs from Wikipedia.")
csv_file = open('../output/classify/racial_terms.csv', 'w', newline='') #'w' write mode
csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
csv_writer.writerow(racist_terms)
csv_file.close()

print("Downloading contemporary ethnic terms from Wikipedia.")
page = requests.get("https://en.wikipedia.org/wiki/List_of_contemporary_ethnic_groups")
tree = html.fromstring(page.content)

terms = tree.xpath('//td/a/text()')

raw_terms = ["".join([ch for ch in word if ch not in punctuation]).lower() for word in terms]

term_set = [word_tokenize(term) for term in raw_terms]
national_terms = []
for t in term_set:
    for w in t:
        if w[0].isalpha():
            if w not in set(stop_list):
                national_terms.append(w)
print("We have",len(national_terms),"contemporary ethnic terms from Wikipedia.")

csv_file = open('../output/classify/national_terms.csv', 'w', newline='') #'w' write mode
csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
csv_writer.writerow(national_terms)
csv_file.close()
"""
print("Loading term lists...")
national_terms = open("../output/classify/national_racial_term_lists/national_terms.csv","r").read()
national_terms = national_terms.split(",")
random.shuffle(national_terms)
racist_terms = open("../output/classify/national_racial_term_lists/racial_terms.csv","r").read()
racist_terms = racist_terms.split(",")
additional_racist_terms = open("../output/classify/national_racial_term_lists/additional_racial_terms.csv","r").read()
additional_racist_terms = additional_racist_terms.split(",")
racist_terms = set(racist_terms + additional_racist_terms)

national_terms = national_terms[:len(racist_terms)]
#print("Loaded", len(racist_terms), "racist terms")
#print("Loaded", len(national_terms), "national terms")
"""
print("Building corpus file name list...")
file_list = []
corpus_path = "../data/text_corrected/"
for path, subdirs, files in os.walk(corpus_path): #change depending on system
    for file in files:
        a = os.path.join(file)
        file_list.append(a)
"""
#file_list = sorted(file_list) #set the range of files to process here
corpus_path = "../data/" # for testing purposes
file_list = "1987_all.txt" # for testing purposes

def training_set_builder(file_name):
    start = time.time()
    print("Process", os.getpid(),"working on file", file_name)
    with open(corpus_path+file_name,encoding='utf-8') as f:
        text = f.read()
    sent_tokens = sent_tokenize(text)
    racialized_sentences = []
    national_sentences = []
    for sent in sent_tokens:
        word_tokens = word_tokenize(sent)
        for word in word_tokens:
            if word in racist_terms:
                loc = sent.find(word)
                if sent[loc-200:loc+200] != "":
                    racialized_sentences.append((sent[loc-200:loc+200],'racial'))
                    #print("Found",word+"! Preview:",sent[loc-100:loc+100]+'\n\n')
            elif word in national_terms:
                loc = sent.find(word)
                if sent[loc-200:loc+200] != "":
                    national_sentences.append((sent[loc-200:loc+200],'national'))
                    #print("Found",word+"! Preview:",sent[loc-100:loc+100]+'\n\n')
    tagged_sentences = racialized_sentences + national_sentences
    #print("number ot tagged sentences ",len(tagged_sentences))
    random.shuffle(tagged_sentences)
    tagged_sentences = tagged_sentences[:20000] #just take the first 20 thousand to keep it small enough
    #pickle results
    pickle_training_set = open("../output/classify/pickled_training_set/"+file_name+"_training_set.pickle","wb")
    pickle.dump(tagged_sentences,pickle_training_set)
    pickle_training_set.close()
    end = time.time()
    print("Time for job, id", os.getpid(),", to complete: {0:.2f}".format(end - start))
    return

def corpus_builder(file_name):
    start = time.time()
    print("Process", os.getpid(),"working on file", file_name)
    with open(corpus_path+file_name,encoding='utf-8') as f:
        text = f.read()
        corpus_list = []
        for word in word_tokenize(text):
            corpus_list.append(word)
    corpus_freqs = nltk.FreqDist(corpus_list)
    #print(corpus_freqs.most_common(100))
    #print(corpus_freqs["slave"])
    corpus_features = list(corpus_freqs.keys())[20:4000]
    #pickle results
    pickle_corpus_freqs = open("../output/classify/pickled_corpus_freqs/"+file_name+"_corpus_freqs.pickle","wb")
    pickle.dump(corpus_features,pickle_corpus_freqs)
    pickle_corpus_freqs.close()
    end = time.time()
    print("Time for job, id", os.getpid(),", to complete: {0:.2f}".format(end - start))
    return

def classifier(file_name):
    start = time.time()
    print("Process", os.getpid(),"working on file", file_name)

    def define_features(text):
        input_doc = word_tokenize(text)
        words = set(input_doc)
        features = {}
        for word in corpus_features:
            features[word] = (word in words)
        return features
    
    corpus_features_pickle = open("../output/classify/pickled_corpus_freqs/"+file_name+"_corpus_freqs.pickle", "rb")
    corpus_features = pickle.load(corpus_features_pickle)
    
    training_set_pickle = open("../output/classify/pickled_training_set/"+file_name+"_training_set.pickle", "rb")
    tagged_sentences = pickle.load(training_set_pickle)

    print("Building featuresets...")
    featuresets = [(define_features(sent), category) for (sent, category) in tagged_sentences]

    corpus_features_pickle.close()
    training_set_pickle.close()

    training_set = featuresets[:2000]
    testing_set = featuresets[2000:4000]
    # posterior = prior occurances x likihood / current evidence
    print("Training Naive Bayes...")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    print("Determining accuracy...")
    print("Naive Bayes accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
    print("Finding most informative features...")
    classifier.show_most_informative_features(15)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

    #NuSVC_classifier = SklearnClassifier(NuSVC())
    #NuSVC_classifier.train(training_set)
    #print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

    #voting_classifier = VotingClassifier(classifier, LinearSVC_classifier,SGDClassifier_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier)

    #print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voting_classifier, testing_set))*100)

    print("Saving classifiers...")
    
    pickle_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier.pickle","wb")
    pickle.dump(classifier,pickle_classifier)
    pickle_classifier.close()
    
    save_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_MNB.pickle","wb")
    pickle.dump(MNB_classifier, save_classifier)
    save_classifier.close()

    save_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_BernoulliNB.pickle","wb")
    pickle.dump(BernoulliNB_classifier, save_classifier)
    save_classifier.close()

    save_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_LR.pickle","wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()

    save_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_LSVC.pickle","wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()

    save_classifier = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_SGDC.pickle","wb")
    pickle.dump(SGDClassifier, save_classifier)
    save_classifier.close()

    end = time.time()
    print("Time for job, id", os.getpid(),", to complete: {0:.2f}".format(end - start))
    return

def racial_classifier(file_name, text):
    open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier.pickle", "rb")
    classifier = pickle.load(open_file)
    open_file.close()

    open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_MNB.pickle", "rb")
    MNB_classifier = pickle.load(open_file)
    open_file.close()

    open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_BernoulliNB.pickle", "rb")
    BernoulliNB_classifier = pickle.load(open_file)
    open_file.close()

    open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_LR.pickle", "rb")
    LogisticRegression_classifier = pickle.load(open_file)
    open_file.close()

    open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_LSVC.pickle", "rb")
    LinearSVC_classifier = pickle.load(open_file)
    open_file.close()

    #open_file = open("../output/classify/pickled_classifiers/"+file_name+"_stored_classifier_SGDC.pickle", "rb")
    #SGDClassifier = pickle.load(open_file)
    #open_file.close()

    corpus_features_pickle = open("../output/classify/pickled_corpus_freqs/"+file_name+"_corpus_freqs.pickle", "rb")
    corpus_features = pickle.load(corpus_features_pickle)
    corpus_features_pickle.close()

    voting_classifier = VotingClassifier(classifier,LinearSVC_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier)

    def define_features(text):
        input_doc = word_tokenize(text)
        words = set(input_doc)
        features = {}
        for word in corpus_features:
            features[word] = (word in words)
        return features
        
    feats = define_features(text)
    return voting_classifier.classify(feats),voting_classifier.confidence(feats)


#######################################
#                                     #
#     Multiprocessing Pool Jobs       #
#                                     #
#######################################

"""
print("Building training sets...")
# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(training_set_builder, file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))

print("Collecting corpora...")
# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(corpus_builder, file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))

print("Running Classifier...")
# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(classifier, file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))
"""
#######################################
#                                     #
#    Interactive Racial Classifier    #
#                                     #
#######################################
#text = ""
#while text != "quit":
#    text = input(">>> ")
#    file_name = file_list
#
#    print(racial_classifier(file_name,text))

#######################################


#########################################
#                                       #
# Individual function calls for testing #
#                                       #
#########################################

#print("Building training sets...")
#training_set_builder(file_name)

#print("Building corpus freqs...")
#corpus_builder(file_name)

#print("Attempting classification...")
#classifier(file_name)

#######################################
