import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords (load once for efficiency)
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords_set and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()


tfidf, model = load_model()

# Streamlit UI
st.title("📱 Email/SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not!")

# Text input
input_sms = st.text_area("Enter the message:", placeholder="Type your message here...")

# Prediction button
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify!")
    else:
        try:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0]

            # 4. Display results
            if result == 1:
                st.error("🚨 **SPAM**")
                st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.success("✅ **NOT SPAM**")
                st.write(f"Confidence: {probability[0]:.2%}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Add footer
st.markdown("---")
st.markdown("*Built with Streamlit • Deployed on Render*")
