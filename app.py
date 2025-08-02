import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Enhanced NLTK downloads for Render deployment
@st.cache_resource
def download_nltk_data():
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # New requirement
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        return False

# Download NLTK data
download_nltk_data()

# Initialize stemmer and stopwords
ps = PorterStemmer()

@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except:
        # Fallback if stopwords download fails
        return set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'])

stopwords_set = get_stopwords()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize with fallback
    try:
        text = nltk.word_tokenize(text)
    except:
        # Simple fallback tokenization if NLTK fails
        text = text.split()

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
            
            # Check if model has predict_proba method
            try:
                probability = model.predict_proba(vector_input)[0]
                confidence_available = True
            except:
                confidence_available = False
            
            # 4. Display results
            if result == 1:
                st.error("🚨 **SPAM**")
                if confidence_available:
                    st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.success("✅ **NOT SPAM**")
                if confidence_available:
                    st.write(f"Confidence: {probability[0]:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Add footer
st.markdown("---")
st.markdown("*Built with Streamlit • Deployed on Render*")

