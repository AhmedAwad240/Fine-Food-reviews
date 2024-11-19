import re
import random
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the saved Tokenizer and OneHotEncoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('encoder.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

# Load the pre-trained model
model = load_model('my_model.h5')

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return ' '.join(lemmatized_tokens)

# Sample sentences for random generation
sample_sentences = [
    "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.",
    "Arrived in 6 days and were so stale i could not eat any of the 6 bags!!",
    "this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault",
    "No tea flavor at all. Just whole brunch of artifial flavors. It is not returnable. I wasted 20+ bucks.",
    "This is what you get in the store",
    "Sad outcome",
    "If you can't handle caffeine, this is not for you.",
    "Yum, Yummy, Yummier"
    'Great for Gluten-free lifestyle!!'
    'Sugar in the raw'
    'Love Love Love'
    'The product is great but the price is out of line'
    'Swedish Pearl is not the same as  Belgian Pearl'
    'Perfect!'
    'great taste'
    'Taste-tested by a wine maker'
    'Excellent Everyday Olive Oil'
    'Love Weavers, I am a fan.'
    'Make My Day'
    'Treat yourself to the best coffee!'
    'Bitter'
    'Drinking it now, love the latin america "aroma"'
    'GREAT SNACK'
    'Best Bar'
    'Cant find anywhere else!'
    'My New Granola Bar'
    'Another Husband Favorite'
    'Price surprise'
    'Very Smooth Coffee - Highly Recommended'
    'A saving grace for Green Mountain Coffee...'
    'My favorite'
    'Good Coffee'


]

# Streamlit app
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis App")
st.write("### Enter a sentence below or generate a random sentence to analyze its sentiment:")

# Button to generate random sentence
if st.button("Generate Random Sentence"):
    random_sentence = random.choice(sample_sentences)
    st.session_state.generated_sentence = random_sentence
    st.success(f"Generated Sentence: **{random_sentence}**")
else:
    random_sentence = st.session_state.get('generated_sentence', '')

# Text input for user
user_input = st.text_area("Your Sentence", random_sentence)

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the new text
        X_new_preprocessed = preprocess_text(user_input)

        # Convert preprocessed text to sequences
        X_new_seq = tokenizer.texts_to_sequences([X_new_preprocessed])
        
        # Pad the sequences to match the input shape expected by the model
        maxlen = 100  # This should match the padding length used during training
        X_new_pad = pad_sequences(X_new_seq, padding='post', maxlen=maxlen)

        # Make predictions with the model
        predictions = model.predict(X_new_pad)

        # Get the index of the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)

        # Map the predicted class to labels
        class_labels = ['Negative', 'Neutral', 'Positive']
        predicted_label = class_labels[predicted_class[0]]

        # Display the predicted label
        st.success(f"Predicted label: **{predicted_label}**")
    else:
        st.warning("Please enter a sentence for prediction.")