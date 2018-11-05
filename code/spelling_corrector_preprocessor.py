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

print("Starting Main Preprocessor...")
print("Importing modules...")

import nltk
import re
from collections import Counter
import os
import sys
import multiprocessing
import time

old_path = "../data/text_preprocessed/" #change depending on system
new_path = "../data/text_corrected/"

print("Collecting file names...")
file_list = []
for path, subdirs, files in os.walk(old_path): 
    for file in files:
        a = os.path.join(file)
        file_list.append(a)

new_file_list = []
for path, subdirs, files in os.walk(new_path): 
    for file in files:
        a = os.path.join(file)
        new_file_list.append(a)

# Spelling Corrector Borrowed from Peter Norvig http://norvig.com/spell-correct.html. Thank you!

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

file_list = [file for file in file_list if file not in set(new_file_list)] # to restart job on specific file

print("Beginning correction of",len(file_list),"files...") 

def processor(file):
    start = time.time()
    print("Process", os.getpid(),"working on file", file)
    enbrit_string = open(old_path+file, "r", encoding="utf-8").read()
    print("Removing common OCR errors and numbers on process", os.getpid(),"for file",file)
    punct_stop_set = {'«','»','/','\\','>','<',')','(','!','+','=','^','*','%','#','@',"'",'"','`','~','■','•','|','[',']','1','2','3','4','5','6','7','8','9','0'}
    enbrit_string = "".join([ch.replace('º','o').replace('ſ','s').replace('- ','') for ch in enbrit_string if ch not in punct_stop_set]) #remove hyphenation and common ocr errors
    print("Tokenizing on process",os.getpid(),"for file",file)
    enbrit_word_tokens = nltk.word_tokenize(enbrit_string.lower())
    #print("Lower casing and removing non-word tokens on process", os.getpid())
    #enbrit_word_tokens_lower = [word.lower() for word in enbrit_word_tokens if word[0].isalpha()]
    print("Removing contractions on process", os.getpid(),"for file",file)
    replacement_patterns = [
        (r'won\'t',r'will not'),
        (r'can\'t',r'cannot'),
        (r'i\'m',r'i am'),
        (r'ain\'t',r'is not'),
        (r'(\w+)\'ll', r'\g<1> will'),
        (r'(\w+)n\'t', r'\g<1> not'),
        (r'(\w+)\'ve', r'\g<1> have'),
        (r'(\w+)\'s', r'\g<1> is'),
        (r'(\w+)\'re', r'\g<1> are'),
        (r'(\w+)\'d', r'\g<1> would'),
        ]
    def replace(text):
        s = text
        for (pattern,repl) in replacement_patterns:
                s = re.sub(pattern, repl, s)
        return s
    enbrit_word_tokens = [replace(word) for word in enbrit_word_tokens]
    #print("Correcting spelling on process", os.getpid(),"for file", file)
    #enbrit_word_tokens = [correction(word) for word in enbrit_word_tokens]
    print("Rebuilding string on process", os.getpid(),"for file",file)
    string_normalized = " ".join(enbrit_word_tokens)
    string_normalized = string_normalized.replace(' .','.').replace(' ;',';').replace(' ,',',').replace(' :',':').replace('- ','').replace(" ’ s","’s" )
    print("Saving process", os.getpid(),"for file",file)
    file_new = open(new_path+file, "w")
    file_new.write(string_normalized)
    file_new.close()
    print("Process", os.getpid(),"done processing", file+"!")
    end = time.time()
    print("Time", os.getpid()," to complete: {0:.2f}".format(end - start))

# capture time start
jobstart = time.time()

# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 7 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(processor, file_list)

# capture close
jobend = time.time()
# print time to complete
print("Time to complete: {0:.2f}".format(jobend - jobstart))
