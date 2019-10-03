import pandas as pd
from spacy.lang.en import English
from collections import Counter
import pickle
from multiprocessing import Pool
from itertools import repeat

def extract_word_freq(doc, word_freq_counter, noun_freq_counter):
    # all tokens that arent stop words or punctuations
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(words)
    word_freq_counter = word_freq_counter + word_freq

    # noun tokens that arent stop words or punctuations
    nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos == "NOUN"]
    noun_freq = Counter(nouns)
    noun_freq_counter = noun_freq_counter + noun_freq

    return [word_freq_counter, noun_freq_counter]


# function for getting word frequencies
def doc_extract_word_freq(docs):
    all_word_freq = Counter()
    all_noun_freq = Counter()
    pool = Pool(processes=1000)
    return pool.starmap(extract_word_freq, zip(docs, repeat(all_word_freq), repeat(all_noun_freq)))


# Load the filtered data set
df = pd.read_csv("data/interim/filtered_reddit_scrape.csv", low_memory=False)

# View distribution of subreddit conversational pairs
print(df['subreddit'].value_counts())
print(df['subreddit'].value_counts(normalize=True))


def create_pairs_list(row):
    pair = str(row['body']) + '|||' + str(row['comment_body'])
    return pair


# create corpus for entire dataset
pairs_list = df.apply(create_pairs_list, axis=1)
print(len(pairs_list))

print(type(pairs_list))

# spacy
nlp = English()

# pipe the pairs to docs
docs = list(nlp.pipe(pairs_list, n_threads=40))

# call the above function to get word and noun frequencies
word_noun_freq = doc_extract_word_freq(docs)

# Pickle the frequencies so that they can be viewed in a Jupyter Notebook
pickle.dump(word_noun_freq, open("data/interim/word_noun_freq.p", "wb"))

print("Script finished!")






