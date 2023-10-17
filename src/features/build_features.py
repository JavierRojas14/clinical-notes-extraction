import re
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")  # Download the Punkt tokenizer data


def preprocess_text(text, language="spanish"):
    # Fill NaN values with an empty string
    text = text.fillna("")

    # Convert text to lowercase
    text = text.str.lower()

    # Remove newlines and extra whitespaces
    text = text.str.replace("\s+", " ", regex=True)

    # Remove punctuation
    text = text.str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)

    # Tokenize the text into words using word_tokenize
    words = text.apply(word_tokenize)

    # Filter out stopwords based on the specified language
    stop_words = set(stopwords.words(language))
    words = words.apply(lambda word_list: [word for word in word_list if word not in stop_words])

    # Join the processed words back into a single string
    processed_text = words.str.join(" ")

    return processed_text
