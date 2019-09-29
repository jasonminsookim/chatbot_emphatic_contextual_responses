#!/usr/bin/env python
# coding: utf-8

# # 1. Libraries and Configuration

# In[1]:


import configparser
import pandas as pd
from datetime import datetime
from psaw import PushshiftAPI
import praw
from tqdm import tqdm

# ## 1.1 Configure PSAW
#
# - __A python wrapper for pushshift.io__

# In[2]:


api = PushshiftAPI()

# ## 1.2 Configure PRAW

# In[3]:


config = configparser.ConfigParser()
config.read('../../praw_config.ini')

r_praw = praw.Reddit(client_id=config['praw_credentials']['client_id'],
                     client_secret=config['praw_credentials']['secret'],
                     redirect_uri='http://localhost:8080',
                     user_agent='chatbot')

# # 3. Scrape Submissions and Comments

# In[4]:


# Subreddits to scrape
subreddits = ['depression', 'anxiety', 'affirmations', 'BodyAcceptance', 'OCD']

# Initialize dictionary for storing scraped data
scraped_dict = {'author': [],
                'score': [],
                'created': [],
                'subreddit': [],
                'title': [],
                'body': [],
                'id': [],
                'comment_author': [],
                'comment_body': [],
                'comment_score': [],
                'comment_edited': []}

# In[5]:


for subreddit in subreddits:
    posts = list(api.search_submissions(limit=10000, subreddit=subreddit, sort='desc', sort_type='score'))

    for post in tqdm(posts):

        # scrape comment data
        submission = r_praw.submission(id=post.id)
        submission.comments.replace_more(limit=0)

        for comment in submission.comments:
            # scrape submission specific data
            scraped_dict['author'].append(post.author)
            scraped_dict['score'].append(post.score)
            scraped_dict['created'].append(post.created_utc)
            scraped_dict['subreddit'].append(post.subreddit)
            scraped_dict['title'].append(post.title)
            try:
                scraped_dict['body'].append(post.selftext)
            except:
                scraped_dict['body'].append(None)
            scraped_dict['id'].append(post.id)

            # scrape comment data
            scraped_dict['comment_author'].append(comment.author)
            scraped_dict['comment_body'].append(comment.body)
            scraped_dict['comment_score'].append(comment.score)
            scraped_dict['comment_edited'].append(comment.edited)

    scraped_df = pd.DataFrame(scraped_dict)
    scraped_df.to_csv('../data/scraped_data/reddit_submission_comments_iter.csv')

# In[6]:


scraped_df = pd.DataFrame(scraped_dict)

# # 4. Wrangle Data
#
# ## 4.1 Convert 'created' from unix time to datetime object

# In[7]:


scraped_df['created'] = scraped_df['created'].apply(
    lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

# In[8]:


print(scraped_df['subreddit'].value_counts())

# In[9]:


scraped_df.to_csv('../../data/raw/scraped_reddit_submissions_comments.csv')

# In[10]:



