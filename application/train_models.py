## Run this to save models locally and then load them directly for inference

import pandas as pd
from tweet_feature_functions import *
from lda_topics import *
from rf_prediction import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
# train and save models to pickle file

###############
## Data Load
print("Loading data...")
input_csv = '../data/twitter_sentiment_data.csv'
df = pd.read_csv(input_csv)

## Basic feature generation
df = dataframe_train_preprocess(df)

###############
## LDA Topics
print("LDA...")

# Preprocess the tweets for LDA
df['processed_message'] = df['message'].apply(lda_preprocess_text)

# parameters
num_topics = 4 # number of topics to find
max_features = 5000 # max features parameter

# calibrate lda
count_vectorizer, lda=lda_train(df, num_topics, max_features)

# save lda models
print("Saving LDA...")
joblib.dump(lda, 'lda_model.pkl')
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')

# generate lda topics
df=apply_lda(df, count_vectorizer, lda, num_topics)

###############
## Train Random Forest
print("Training Random Forest...")

# feature set for training
feature_set=['is_weekend',  #date info
            'message_length','word_count', # length info
            'is_retweet', 'num_mentions','num_hashtags','num_links','num_emojis', #character infp
            'topic_0','topic_1', 'topic_2', 'topic_3'] #topics info

class_descriptions = {
    -1: 'Against',
    0: 'Neutral',
    1: 'Pro',
    2: 'News'
}

# Define the features and target
X = df[feature_set]
y = df['sentiment']
feature_names = X.columns

# Random Forest with Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [20, 50],
    'min_samples_leaf': [10, 20]
}
rf_classifier = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X, y)
best_rf_classifier = grid_search.best_estimator_

# Save RF model
print("Saving RF model...")
joblib.dump(best_rf_classifier, 'best_rf_classifier.pkl')

print("Training Finalized!")