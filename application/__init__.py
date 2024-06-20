from flask import Flask, request, Response, json, jsonify
import pandas as pd
from application.tweet_feature_functions import *
from application.lda_topics import *
from application.rf_prediction import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


###############
## Data Load
print("Loading data...")
input_csv = './data/twitter_sentiment_data.csv'
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

print('Training finished!')
# predict
#rf_predictions = best_rf_classifier.predict(X)

################################
#create flask instance (this is for deployment)

app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST']) # GET: can cache data but POST is more secure.
def predict():
    
    #get data from request
    data = request.get_json(force=True) # user input from web app
    
    #data = np.array([data['date'],data['text']]) # convert to np array
    #print("Data from request:",data)

    # Convert the JSON data to a DataFrame 
    df = pd.DataFrame([data])
    print("Dataframe:\n", df)
    
    # generate basic features
    df = dataframe_live_preprocess(df)
    
    # Preprocess the tweets for LDA
    df['processed_message'] = df['message'].apply(lda_preprocess_text)
    
    # generate lda topics
    df=apply_lda(df, count_vectorizer, lda, num_topics)
    
    print("Preprocessed data:\n",df)

    X=df[feature_set]

    predicted_class, predicted_class_probability, predicted_class_description, class_probabilities = rf_prediction(X, best_rf_classifier, class_descriptions)
    
    # Collect feature values for response
    features = {feature: df.at[0, feature] for feature in feature_set}
    # Format topic features as percentages
    for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3']:
        features[topic] = f"{features[topic] * 100:.2f}%"

    # Convert all feature values to strings
    features = {feature: str(value) for feature, value in features.items()}

    # Return a structured JSON response
    response = {
        'predicted_class': predicted_class,
        'predicted_class_description': predicted_class_description,
        'predicted_class_probability': f"{predicted_class_probability * 100:.2f}%",
        'class_probabilities': class_probabilities,
        'features': features
    }

    return jsonify(response)
    #return Response(json.dumps(predicted_class)) # convert to json back to responde object