import streamlit as st
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load model and vectorizer
loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# Function to process and predict
def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]
    input_data = [' '.join(corpus)]
    vectorized_input = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Check if the news you entered is real or fake.")

# Input box
user_input = st.text_area("Enter the News Article or Headline:")

if st.button("Check"):
    if user_input.strip() != "":
        result = fake_news_det(user_input)
        if result == 1:
            st.error("🔴 Prediction: Looks like **Fake News**!")
        else:
            st.success("🟢 Prediction: Looks like **Real News**.")
    else:
        st.warning("⚠️ Please enter some text to check.")
