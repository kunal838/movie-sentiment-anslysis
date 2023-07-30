import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load the trained sentiment analysis model
model = tf.keras.models.load_model('mini.h5')

# Function to preprocess the review
def preprocess_review(review, word_index):
    max_sequence_length = 200
    review_seq = [word_index[word] if word in word_index else 0 for word in review.lower().split()]
    review_seq = pad_sequences([review_seq], maxlen=max_sequence_length)

    return review_seq

# Function to predict sentiment
def predict_sentiment(review):
    word_index = imdb.get_word_index()
    review_seq = preprocess_review(review, word_index)

    # Make prediction using the model
    prediction = model.predict(review_seq)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    return sentiment

# Main Streamlit app
def main():
    st.title("Movie Review Sentiment Analysis")
    st.markdown(
        """
        <style>
        body {
            color: #FFFFFF;
            background-color: #1E1E1E;
        }
        .stTextInput > div > div > input {
            background-color: #272727;
            color: #FFFFFF;
        }
        .stButton button {
            background-color: #4CAF50;
            color: #FFFFFF;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    review_input = st.text_area("Enter your movie review:")
    if st.button("Analyze Sentiment"):
        if review_input.strip():
            sentiment = predict_sentiment(review_input)
            st.write(f"Sentiment: {sentiment}")
        else:
            st.warning("Please enter a movie review.")

if __name__ == '__main__':
    main()
