import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import re
import pickle

df = pd.read_csv("data/interim/filtered_reddit_scrape.csv", low_memory=False)


# clean the text and replace concatenated words.
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"\n", "", text)

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    # text = re.sub(r"([?.!,])", r" \1 ", text)

    return text


prompts = df['body'].apply(clean_text)
responses = df['comment_body'].apply(clean_text)

# tokenize

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    prompts + responses, target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


tokenized_prompts, tokenized_responses = tokenize_and_filter(prompts, responses)

# save constant values
constants = {'START_TOKEN': START_TOKEN, 'END_TOKEN': END_TOKEN, 'VOCAB_SIZE': VOCAB_SIZE}
pickle.dump(constants, open("data/processed/constants.p", "wb"))  # pickle prompts

# save the tokenized prompts and responses
pickle.dump(tokenized_prompts, open("data/processed/tok_prompts.p", "wb"))  # pickle prompts
pickle.dump(tokenized_responses, open("data/processed/tok_responses.p", "wb"))  # pickle prompts
