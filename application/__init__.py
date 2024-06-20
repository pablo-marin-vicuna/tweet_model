from flask import Flask, request, Response, json, jsonify
import pandas as pd
import numpy as np
import joblib 

app = Flask(__name__)

#############
## Train and save models to pickle files
# python train_models.py

###############
## Load pre-trained models
best_rf_classifier = joblib.load('application/best_rf_classifier.pkl')
lda = joblib.load('application/lda_model.pkl')
count_vectorizer = joblib.load('application/count_vectorizer.pkl')

from application.tweet_feature_functions import *
from application.lda_topics import *
from application.rf_prediction import *

num_topics=4

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

################################
#create flask instance (this is for deployment)

app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST']) # GET: can cache data but POST is more secure.
def predict():
    
    #get data from request
    data = request.get_json(force=True) # user input from web app
    
    # Convert the JSON data to a DataFrame 
    df = pd.DataFrame([data])
    print("Dataframe:\n", df)
    
    # generate basic features
    df = dataframe_live_preprocess(df)
    
    # Preprocess the tweets for LDA
    df['processed_message'] = df['message'].apply(lda_preprocess_text)
    
    # generate lda topics
    df=apply_lda(df, count_vectorizer, lda, num_topics=num_topics)
    
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