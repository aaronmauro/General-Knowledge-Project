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

print("Starting Sentiment Analyzer")
import nltk
import os
import sys
import csv
import time
import multiprocessing
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation, extract_unigram_feats
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print("Finished Imports\n\nDetermining File Paths")

print("Collecting file names...")
file_list = []
corpus_path = "../data/text_corrected/"
for path, subdirs, files in os.walk(corpus_path): #change depending on system
    for file in files:
        a = os.path.join(file)
        file_list.append(a)
#print(file_list)
print("Training Classifier")
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)

train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
print("Number of unigram features: ",len(unigram_feats))

sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

file_list = file_list # select file ranges here

print("Beginning Sentiment Classification of",len(file_list),"files...") 

sid = SentimentIntensityAnalyzer()

def sentiment_classifier(file_name):
    start = time.time()
    text = open(corpus_path+file_name,encoding='utf-8').read()
    print("Process", os.getpid(),"working on file", file_name)
    sent_tokens = nltk.sent_tokenize(text)
    csv_file = open('../output/sentiment/'+file_name[:-4]+'_sentiment.csv', 'w', newline='') #'w' write mode
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    csv_writer.writerow(['File Name', 'Year','Pos','Neg','Compound','sentence'])
    csv_file.close()
    
    for sentence in sent_tokens:
        #print(sentence)
        ss = sid.polarity_scores(sentence)
        csv_file = open('../output/sentiment/'+file_name[:-4]+'_sentiment.csv', 'a', newline='') # 'a' append mode
        csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
        csv_writer.writerow([file[:-4],file[:4],ss['pos'],ss['neg'], ss['compound'],sentence])
        # to print pos, neg, compound results from dictionary as single line
        #for k in sorted(ss):
        #    csv_writer.writerow([file[:-4],file[:4], k, ss[k]])
        csv_file.close()

    end = time.time()
    print("Time for job, id", os.getpid(),", to complete: {0:.2f}".format(end - start))
    return

# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(sentiment_classifier, file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))