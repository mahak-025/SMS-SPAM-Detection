import nltk # type: ignore
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st # type: ignore
import pickle 
import string
from nltk.corpus import stopwords # type: ignore
from nltk.stem.porter import PorterStemmer # type: ignore
import pickle
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgb(189, 229, 236); /* Change this color to your preference */
    }
    
    .stButton > button {
        color: white;
        background-color:rgb(64, 110, 161);
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


file_path = r"C:\Users\DELL\SMS-SPAM-Detection\vectorizer.pkl"

try:
    with open(file_path, "rb") as file:
        tf = pickle.load(file)
    print("vectorizer.pkl loaded successfully!")
except FileNotFoundError:
    print(f"File not found at: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")



# Load the trained model
model_path = r"C:\Users\DELL\SMS-SPAM-Detection\model.pkl"

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

    


    
ps = PorterStemmer()


stopwords = stopwords.words("English")
def transform_text(text):
    usefull_words = []
    text = text.lower()
    words = nltk.word_tokenize(text)
    for word in words:
        if word not in stopwords:
            usefull_words.append(word)
    print(usefull_words)
    return " ".join(usefull_words)

st.title("SMS Spam Detection Model")
st.write(" by Mahak Chouhan")
    

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tf.transform([transformed_sms])

        # Predict
    
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
