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

# Terms scanned from Irving Lewis Allen's "The Word List" in The Language of Ethnic Conflict: Social Organization and Lexical Culture (Columbia University Press) 1983. pp. 45-72.

import nltk
from string import punctuation
import csv
text = open("../data/allen_the_language_of_ethnic_conflict.txt").read()
punctuation = punctuation + "“" + "1234567890"
punct = [p for p in punctuation if p is not "-"]
text_no_punct = "".join([ch for ch in text if ch not in punct])
tokens = nltk.word_tokenize(text_no_punct.lower())
#print(len(tokens))
tokens_set = set(tokens)
#print(len(tokens_set))
scrabble = open("../data/scrabble_word_list.txt").read()
scrabble_tokens = nltk.word_tokenize(scrabble)
national_terms = open("../output/classify/national_racial_term_lists/national_terms.csv","r").read()
national_terms = national_terms.split(",")
ethnic_slurs = [word for word in tokens_set if word not in scrabble_tokens and word not in national_terms and len(word) > 3]
#print(ethnic_slurs)
#print(len(ethnic_slurs))
csv_file = open('../output/classify/additional_racial_terms.csv', 'w', newline='') #'w' write mode
csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
csv_writer.writerow(ethnic_slurs)
csv_file.close()