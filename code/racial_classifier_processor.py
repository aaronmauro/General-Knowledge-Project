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
import pickle
import os
import sys
import csv
import time
import multiprocessing
from racial_classifier import racial_classifier
from nltk import sent_tokenize, word_tokenize
from term_set import term_set

term_set = term_set()

print("Collecting file names...")
corpus_path = "../data/text_corrected/"
new_path = "../output/classify/results/"

file_list = []
for path, subdirs, files in os.walk(corpus_path): #change depending on system
    for file in files:
        a = os.path.join(file)
        file_list.append(a)

new_file_list = []
for path, subdirs, files in os.walk(new_path): #change depending on system
    for file in files:
        a = os.path.join(file)
        new_file_list.append(a[:-13]+".txt")

new_file_list = set(new_file_list)

file_list = [file for file in file_list if file not in new_file_list]
print("Processing",len(file_list),"remaining files")

def racial_classifier_processor(file_name):
    start = time.time()
    text = open(corpus_path+file_name,encoding='utf-8').read()
    print("Process", os.getpid(),"working on file", file_name)
    sent_tokens = sent_tokenize(text)
    csv_file = open('../output/classify/results/'+file_name[:-4]+'_classify.csv', 'w', newline='') #'w' write mode
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    csv_writer.writerow(['File Name', 'Year','result','score','sentence'])
    csv_file.close()
    
    csv_file = open('../output/classify/results/'+file_name[:-4]+'_classify.csv', 'a', newline='') # 'a' append mode
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    sent_group = []
    for sentence in sent_tokens:
        found = 0
        if len(sent_group) < 7: # calibrate for larger sentence chunks
            sent_group.append(sentence)
            tokens = word_tokenize("".join(sent_group))
            for token in tokens:
                if token in term_set:
                    found += 1
                else:
                    found += 0
            if found > 5: # calibrate if too many false positives occuring
                result = str(racial_classifier("1987_vol.1.txt","".join(sent_group)))
                csv_writer.writerow([file_name[:-4],file_name[:4],result[2:8],result[11:-1],"".join(sent_group)])
                sent_group = []
            else:
                result = "('neutral', 0.0)"
                csv_writer.writerow([file_name[:-4],file_name[:4],result[2:-7],result[11:-1],"".join(sent_group)])
                sent_group = []
    csv_file.close()
    end = time.time()
    print("Time for job, id", os.getpid(),", to complete: {0:.2f}".format(end - start))
    return

# racial_classifier_processor("1987_vol.1.txt") #for testing

# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=8) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(racial_classifier_processor, file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))
