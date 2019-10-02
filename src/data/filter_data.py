import pandas as pd

# read csv
df = pd.read_csv('data/raw/scraped_reddit_submissions_comments.csv', low_memory=False)

# remove unnecessary column
df = df.drop(['Unnamed: 0'], axis=1)

# remove comments that were deleted or removed
filt_df = df[~df['comment_body'].isin(['[removed]', '[deleted]'])]

# remove submissions that were deleted or removed
filt_df = filt_df[~filt_df['body'].isin(['[removed]', '[deleted]'])]

# save the filtered data set as a csv
filt_df.to_csv("data/interim/filtered_reddit_scrape.csv", index=False)
