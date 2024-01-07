import re
import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import os

# Set the working directory to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the Naive Bayes model
model_naive = joblib.load('naive_bayes_model.joblib')

# Load other necessary components
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
stemmer = PorterStemmer()

# Load or preprocess the data used to train the model
training_data = pd.read_csv('Ukraine_10K_tweets_sentiment_analysis.csv')

# Replace NaN values in 'Tweet' column with an empty string
training_data['Tweet'] = training_data['Tweet'].fillna('')

tidy_tweets = training_data['Tweet']

# Fit the vectorizer on the training data
bow_vectorizer.fit(tidy_tweets)

# Function for preprocessing
def preprocess_data(input_text):
    pattern_mentions = r'@[\w]*'
    pattern_angle_brackets = r'<|>'
    tidy_tweet = re.sub(pattern_mentions, '', input_text)
    tidy_tweet = re.sub(pattern_angle_brackets, '', tidy_tweet)
    tidy_tweet = re.sub("[^a-zA-Z]", " ", tidy_tweet)
    tidy_tweet = ' '.join([w for w in tidy_tweet.split() if len(w) > 3])
    
    # Tokenization and stemming
    tokenized_tweet = tidy_tweet.split()
    tokenized_tweet = [stemmer.stem(i) for i in tokenized_tweet]
    tidy_tweet = ' '.join(tokenized_tweet)
    
    return tidy_tweet

# Streamlit app
def main():
    st.title("Sentiment Analysis Web App")

    # User input for prediction
    user_input = st.text_area("Enter a sentence for sentiment analysis:")

    if st.button("Predict Sentiment"):
        if user_input:
            # Preprocess the input
            processed_input = preprocess_data(user_input)
            
            # Transform the processed input using the bow_vectorizer
            input_vector = bow_vectorizer.transform([processed_input])
            
            # Predict sentiment
            predicted_sentiment = model_naive.predict(input_vector)
            
            # Map labels to 'Positive' or 'Negative'
            sentiment_label = "Positive" if predicted_sentiment[0] == 1 else "Negative"
            
            st.success(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.warning("Please enter a sentence for prediction.")

if __name__ == "__main__":
    main()
