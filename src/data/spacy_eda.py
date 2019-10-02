import pandas as pd
import spacy

df = pd.read_csv("data/interim/filtered_reddit_scrape.csv", low_memory=False)


# View distribution of subreddit conversational pairs
print(df['subreddit'].value_counts())
print(df['subreddit'].value_counts(normalize=True))


def create_corpus(df):
    corpus = df['body'] + df['comment_body']
    corpus = corpus.str.cat(sep='|||')
    return corpus


# create corpus for entire dataset
full_corpus = create_corpus(df)

print(len(full_corpus))
# create corpus for each subreddit
subreddit_corpuses = df.apply(create_corpus, axis=1)

# spacy
nlp = spacy.load("en")

# This will not run due to the shear size of the corpus. Instead, try to do batch processing.
# Multi-core processing may help expedite this process
# for subreddit_corpus in subreddit_corpuses:
#     subreddit_doc = nlp(subreddit_corpus)
#
#     for token in subreddit_doc:
#         print(token.text, token.pos_, token.dep_)



