## LDA topic modeling

import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

# Get the existing stop words from nltk
stop_words = set(stopwords.words('english'))

# Add custom stop words to the set
additional_stop_words = ['rt', 'climate', 'change','global', 'warming','trump']
stop_words.update(additional_stop_words)

# preprocessing function
def lda_preprocess_text(message):
    # Lowercase the text
    message = message.lower()
    # Remove punctuation
    message = re.sub(r'[^\w\s]', '', message)
    # Tokenize and remove stopwords
    tokens = message.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


# LDA training
def lda_train(df, num_topics=4, max_features=5000):

    # Create a document-term matrix
    count_vectorizer = CountVectorizer(max_features=max_features)
    dtm = count_vectorizer.fit_transform(df['processed_message'])

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    return count_vectorizer, lda


def apply_lda(df, count_vectorizer, lda, num_topics):
    # Apply to test data set
    dtm = count_vectorizer.transform(df['processed_message'])
    topic_distributions_test = lda.transform(dtm)
    for i in range(num_topics):
        df[f'topic_{i}'] = topic_distributions_test[:, i]

    # Add the most likely topic column
    df['most_likely_topic'] = np.argmax(topic_distributions_test, axis=1)

    return df