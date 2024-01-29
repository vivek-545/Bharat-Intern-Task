import streamlit as st
import pickle
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import PorterStemmer

# Load the TfidfVectorizer and the trained model with error handling
try:
    with open('vectorizer2.pkl', 'rb') as vect_file:
        tfidf = pickle.load(vect_file)

    with open('model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model files not found. Please ensure correct file paths.")

ps = PorterStemmer()

# Text transformation function


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [ps.stem(token) for token in tokens if token.isalnum()]
    return " ".join(stemmed_tokens)

# Function to predict spam probability


def predict_spam_probability(input_text):
    transformed_text = transform_text(input_text)
    vectorized_text = tfidf.transform([transformed_text])
    prob_spam = model.predict_proba(vectorized_text)
    spam_probability = prob_spam[0][1]
    return spam_probability


st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip():
        try:
            spam_probability = predict_spam_probability(input_sms)
            threshold = 0.04  # You can adjust this threshold
            if spam_probability > threshold:
                st.error("The given message is Spam")

            else:
                st.success("The given message is not Spam")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message to predict.")
