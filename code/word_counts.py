import nltk
import pygal
import re
import os
import sys
import csv
import pandas as pd
import multiprocessing
import time
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

text_corrected = "../data/text_corrected/"
results = "../output/word_counts/results/"

file_list = []
for path, subdirs, files in os.walk(text_corrected):
    for file in files:
        a = os.path.join(file)
        file_list.append(a)
file_list = sorted(file_list)

results_file_list = []
for path, subdirs, files in os.walk(results):
    for file in files:
        a = os.path.join(file)
        results_file_list.append(a)
results_file_list = sorted(results_file_list)

stops = set(stopwords.words('english'))

# open data frame in pandas
def open_df(path,file_name):
    df = pd.read_csv(path+file_name, 
                    #index_col='Year', 
                    #parse_dates=['Year'],
                    header=0)
    return df

#open_df(results,"word_counts_by_vol.csv")

#group word counts in tuples with list of publishing dates
def edition_wc():
    years = []
    edition_word_counts = []
    count_group = []
    df = open_df(results,"word_counts_by_vol.csv")
    for index, row in df.iterrows():
        if row['Year'] not in set(years):
            years.append(row['Year'])
    for year in years:
        for index, row in df.loc[df['Year'] == year].iterrows():
            count_group.append(row['Word Count'])
        count_group = tuple(count_group)
        #count_group = sum(count_group) #toggle on and off to get edition totals or volume totals
        edition_word_counts.append(count_group)
        count_group = []
    return years, edition_word_counts

# edition_wc()

def lemmatize(path,file_name):
    with open(path+file_name) as f:
        text = f.read()
        word_tokens = nltk.word_tokenize(text)
        wnlemmatizer = WordNetLemmatizer()
        lemmas = [wnlemmatizer.lemmatize(word) for word in word_tokens if len(word) > 2 and word not in stops]
    return lemmas

#lemmatize(text_corrected,"1771_v.1.txt")

def counts(file_name):
    counts = {}
    words = lemmatize(text_corrected,file_name)
    for w in words:
        counts[w] = counts.get(w,0) + 1
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1])
    csv_file = open('../output/word_counts/results/'+file_name[:-4]+'_word_counts.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    csv_writer.writerow(['File Name', 'Year','word','count'])
    csv_file = open('../output/word_counts/results/'+file_name[:-4]+'_word_counts.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    for w,c in sorted_counts:
        csv_writer.writerow([file_name[:-4],file_name[:4], w,c])
    csv_file.close()
    return

#counts(text_corrected,"1771_v.1.txt")

def graph_counts(file_name):
    df = pd.read_csv('../output/word_counts/results/'+file_name, header=0)
    data = df.tail(100)
    chart = pygal.Line(x_label_rotation=90, width=1000,show_legend=False)
    chart.title = 'Top 100 Most Common Words in '+file_name[:-16]
    words = []
    counts = []
    for index,row in data.iterrows():
        words.append(row['word'])
        counts.append(row['count'])
    chart.add("Word Count",counts[::-1])
    chart.x_labels = words[::-1]
    chart.render_to_file('../output/word_counts/graphs/mfw_'+file_name[:-4]+'.svg')
    return
for file in results_file_list:
    graph_counts(file)


"""
# capture time start
jobstart = time.time()
# begin mutliprocessing
pool = multiprocessing.Pool(processes=7) #default uses as many cores as available (set at 4 core for typical machine)
# map processor function jobs to processor core pool
result = pool.map(graph_counts, results_file_list)
# capture close
jobend = time.time()
# print time to complete
print("Time to complete for all files: {0:.2f}".format(jobend - jobstart))
"""