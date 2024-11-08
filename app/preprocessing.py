import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download all required NLTK data
try:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt_tab")
    nltk.download("omw-1.4")  # Open Multilingual Wordnet
except:
    print("Some NLTK data might not have been downloaded properly")


def preprocess_text(text: str) -> str:
    """
    Preprocess the text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing stopwords
    4. Lemmatizing words
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    try:
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Join back into text
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return text  # Return original text if preprocessing fails
