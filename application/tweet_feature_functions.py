# feature extractions from message
from datetime import *
import re
import emoji

import nltk
from nltk.corpus import stopwords

##################
## Simple functions

# Function to convert tweet ID to datetime
def tweet_id_to_datetime(tweet_id):
    twitter_epoch = 1288834974657
    timestamp = (tweet_id >> 22) + twitter_epoch
    return datetime.fromtimestamp(timestamp / 1000.0)

# function to determine if date is a weekend
def is_weekend(date):
    # Convert the string date to a datetime object
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    # Check if the day is Saturday (5) or Sunday (6)
    return 1 if date_obj.weekday() >= 5 else 0

# Function to identify if a message is a retweet
def is_retweet(message):
    return 1 if message.startswith('RT') else 0

# Function to extract user mentions
def extract_mentions(message):
    return re.findall(r'@\w+', message)

# Function to identify if a hashtag is present
def extract_hashtags(message):
    return re.findall(r'#\w+', message)

# Function to identify if a link is present
def has_link(message):
    return re.findall(r'http[s]?://', message)

# Function to identify if emojis are present and count them
def extract_emojis(message):
    return [char for char in message if emoji.is_emoji(char)]

#################
## NLTK functions

# Tokenize messages and remove stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text: tokenize and remove stop words
def nltk_preprocess(text):
    tokens = text.lower().split()
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens


#################
## Full preprocess function

def dataframe_preprocess(df):
    # Convert tweet ID to datetime
    df['datetime'] = df['tweetid'].apply(tweet_id_to_datetime)

    # Apply the functions to create new columns
    df['is_weekend'] = df['date'].apply(is_weekend)
    df['message_length'] = df['message'].apply(len)

    # Apply NLTK preprocessing to the messages
    df['tokens'] = df['message'].apply(nltk_preprocess)
    df['word_count'] = df['tokens'].apply(len)

    # Tweet content extraction
    df['is_retweet'] = df['message'].apply(is_retweet)

    df['mentions'] = df['message'].apply(extract_mentions)
    df['hashtags'] = df['message'].apply(extract_hashtags)
    df['links'] = df['message'].apply(has_link)
    df['emojis'] = df['message'].apply(extract_emojis)

    # Count the number of mentions, hashtags, links, and emojis
    df['num_mentions'] = df['mentions'].apply(len)
    df['num_hashtags'] = df['hashtags'].apply(len)
    df['num_links'] = df['links'].apply(len)
    df['num_emojis'] = df['emojis'].apply(len)

    return df