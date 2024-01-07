import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

# Load the dataset
twitter = pd.read_csv('Ukraine_10K_tweets_sentiment_analysis.csv')
twitter = twitter.dropna()

# Split data into training and test
X = twitter['Tweet']
y = twitter['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Positive
positive_tweets = twitter[twitter['Sentiment'] == 'Positive']
positive_tweets.head(10)

# Negative
negative_tweets = twitter[twitter['Sentiment'] == 'Negative']
negative_tweets.head(10)

length_train_dataset = X_train.str.len()
length_test_dataset = X_test.str.len()

plt.hist(length_train_dataset, bins=10, label="Train tweets")
plt.hist(length_test_dataset, bins=10, label="Test tweets")
plt.legend()
plt.show()

import pandas as pd

pattern_mentions = r'@[\w]*'  # Define the pattern to remove @mentions
pattern_angle_brackets = r'<|>'  # Define the pattern to remove angle brackets

# Apply the pattern to remove @mentions and angle brackets in 'text' column
twitter['tidy_tweet'] = twitter['Tweet'].str.replace(pattern_mentions, '').str.replace(pattern_angle_brackets, '')
twitter['tidy_tweet'] = twitter['tidy_tweet'].str.replace("[^a-zA-Z]", " ")
twitter['tidy_tweet'] = twitter['tidy_tweet'].fillna('')  # Replace NaN values with empty string
twitter['tidy_tweet'] = twitter['tidy_tweet'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w) > 3]))

#split words
tokenized_tweet = twitter['tidy_tweet'].apply(lambda x:x.split())
tokenized_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
twitter['tidy_tweet'] = tokenized_tweet

# All Sentiment
all_words = ' '.join([text for text in twitter['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=600,height=400,random_state=50,max_font_size=100).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Positive Sentiment
normal_words= ' '.join([text for text in twitter['tidy_tweet'][twitter['Sentiment']=='Positive']])
twitter['Sentiment']
wordcloud = WordCloud(width=600,height=400,random_state=50,max_font_size=100).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative Sentiment

negative_words= ' '.join([text for text in twitter['tidy_tweet'][twitter['Sentiment']=='Negative']])
wordcloud= WordCloud(width=600,height=400,random_state=50,max_font_size=100).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(twitter['tidy_tweet'])

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
twitter['Sentiment'] = label_encoder.fit_transform(twitter['Sentiment'])

#make null values by 0
twitter=twitter.fillna(0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, twitter['Sentiment'], test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
model_naive = MultinomialNB().fit(X_train, y_train)
predicted_naive = model_naive.predict(X_test)
joblib.dump(model_naive, 'naive_bayes_model.joblib')
from sklearn.metrics import confusion_matrix

plt.figure(dpi=200)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix Naive Bayes')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig("confusion_matrix.png")
plt.show()

from sklearn.metrics import accuracy_score

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ",score_naive)

from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_test, predicted_naive)
print(report)

# Fungsi untuk memprediksi sentimen
def predict_sentiment(input_text):
    # Preprocessing teks input
    input_text = re.sub("[^a-zA-Z]", " ", input_text)  # Hapus karakter non-alfabet
    input_text = ' '.join([stemmer.stem(word) for word in input_text.split()])  # Stemming
    input_text = bow_vectorizer.transform([input_text])  # Ubah ke vektor BoW

    # Prediksi sentimen
    prediction = model_naive.predict(input_text)

    # Kembalikan hasil prediksi
    return label_encoder.inverse_transform(prediction)[0]

# Contoh penggunaan
example_positive = "This is ridiculous, this is a good project"
predicted_sentiment = predict_sentiment(example_positive)
print(f"Sentimen prediksi: {predicted_sentiment}")

example_negative = "Ukraine channels finances into a crypto fraud, resulting in a loss of 40."
predicted_sentiment = predict_sentiment(example_negative)
print(f"Sentimen prediksi: {predicted_sentiment}")
