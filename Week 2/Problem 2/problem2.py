import pandas as pd
import numpy as np
import nltk
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 1. Load Data
df_tweets = pd.read_csv("Problem 2/Tweets.csv")[["airline_sentiment", "text"]]
df_tweets.columns = ["Sentiment", "Tweet"]

# 2. Preprocessing
def preprocess_text_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words])

df_tweets["Processed_Tweet"] = df_tweets["Tweet"].apply(preprocess_text_tweet)

# 3. Load Word2Vec
w2v_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# 4. Convert to Vectors
def tweet_to_vector(tweet):
    vectors = [w2v_model[word] for word in tweet.split() if word in w2v_model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

df_tweets["Tweet_Vector"] = df_tweets["Processed_Tweet"].apply(tweet_to_vector)
X_tweets = np.array(df_tweets["Tweet_Vector"].tolist())
y_tweets = df_tweets["Sentiment"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tweets, y_tweets, test_size=0.2, stratify=y_tweets, random_state=42)

# 6. Train and Evaluate
model_tweet = LogisticRegression(max_iter=1000, solver='liblinear', multi_class='ovr')
model_tweet.fit(X_train, y_train)
y_pred = model_tweet.predict(X_test)

print(f"Twitter Sentiment Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict Function
def predict_tweet_sentiment(tweet):
    processed = preprocess_text_tweet(tweet)
    vector = tweet_to_vector(processed).reshape(1, -1)
    return model_tweet.predict(vector)[0]

# Test
print(predict_tweet_sentiment("I love Delta! Best airline."))
print(predict_tweet_sentiment("Delayed for 5 hours. Worst flight ever."))
print(predict_tweet_sentiment("Flight 321 arrived on time."))
