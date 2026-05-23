import streamlit as st
import nltk

# Programmatically download the required NLTK resource
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# If your code also uses NLTK stopwords later on, add this as well:
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
import streamlit as st
import pickle
import string
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Silence unnecessary third-party warnings (like UserWarnings from sklearn/pickle)
warnings.filterwarnings("ignore", category=UserWarning)


# 2. Quietly download NLTK resources without printing verbose download logs
@st.cache_resource
def load_nltk():
    try:
        # Using download with 'quiet=True' prevents NLTK logging warnings
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        pass  # Fallback if already downloaded or offline


load_nltk()

ps = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Use a set for stopwords to fix performance/lookup warnings
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# 3. Load files safely
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Could not find 'vectorizer.pkl' or 'model.pkl'. Please ensure they are in the same directory.")
    st.stop()

# Streamlit UI
st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the Mail")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter some text first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("Spam")
        else:
            st.success("No Spam")

            #streamlit run app.py