import nltk
nltk.data.path.append("C:/Users/Jennifer/AppData/Roaming/nltk_data")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
jenn_w2v_path = "../GoogleNews-vectors-negative300.bin"

import pandas as jenn_pd
import numpy as jenn_np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

jenn_df_sms = jenn_pd.read_csv('Problem 1/spam.csv', encoding='latin-1').iloc[:, :2]
jenn_df_sms.columns = ['Label', 'Message']

def jenn_preprocess_text_sms(jenn_text):
    jenn_text = jenn_text.lower()
    jenn_tokens = word_tokenize(jenn_text)
    jenn_stop_words = set(stopwords.words('english'))
    return " ".join([jenn_word for jenn_word in jenn_tokens if jenn_word.isalnum() and jenn_word not in jenn_stop_words])

jenn_df_sms['Processed_Message'] = jenn_df_sms['Message'].apply(jenn_preprocess_text_sms)

jenn_w2v_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def jenn_message_to_vector(jenn_message):
    jenn_vectors = [jenn_w2v_model[jenn_word] for jenn_word in jenn_message.split() if jenn_word in jenn_w2v_model.key_to_index]
    return jenn_np.mean(jenn_vectors, axis=0) if jenn_vectors else jenn_np.zeros(jenn_w2v_model.vector_size)

jenn_df_sms['Message_Vector'] = jenn_df_sms['Processed_Message'].apply(jenn_message_to_vector)
jenn_X_sms = jenn_np.array(jenn_df_sms['Message_Vector'].tolist())
jenn_y_sms = jenn_df_sms['Label'].apply(lambda jenn_x: 1 if jenn_x == 'spam' else 0)

jenn_X_train, jenn_X_test, jenn_y_train, jenn_y_test = train_test_split(jenn_X_sms, jenn_y_sms, test_size=0.2, random_state=42)

jenn_model_sms = LogisticRegression(max_iter=1000)
jenn_model_sms.fit(jenn_X_train, jenn_y_train)
jenn_y_pred = jenn_model_sms.predict(jenn_X_test)
print(f"SMS Spam Accuracy: {accuracy_score(jenn_y_test, jenn_y_pred):.4f}")

def jenn_predict_message_class(jenn_message):
    jenn_processed = jenn_preprocess_text_sms(jenn_message)
    jenn_vector = jenn_message_to_vector(jenn_processed).reshape(1, -1)
    return 'spam' if jenn_model_sms.predict(jenn_vector)[0] == 1 else 'ham'

print(jenn_predict_message_class("Win a free ticket now!!!"))
print(jenn_predict_message_class("Let's meet tomorrow."))
