#Step 1: Import Libraries and Load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the IMDM DATASET word index
 
word_index = imdb.get_word_index()
reverse_word_index = {(value, key) for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('imdb_simple_rnn_model.h5')

# step 2: Helper Functions
# Function to Decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



import streamlit as st
import pandas as pd
# streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Entre a movie review to classify it as positive or negative.')

#user input 
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_text = preprocess_text(user_input)

    #make prediction
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write(f'please entre a movie review')  