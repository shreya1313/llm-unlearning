import re
import pandas as pd

stopwords = ["the", "and", "is", "on", "in", "if", "for", "a", "an", "of", "or", "to", "it", "you", "your"]

def re_preprocess_text(text):
    """
    Clean and preprocess the input text.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove web links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove special characters, punctuation marks, and newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove stopwords
    text = ' '.join(word for word in text.split() if word.lower() not in stopwords)

    return text.lower()


def preprocess_dataframe(train):
    train['Cleaned_Comments'] = train['comment_text'].apply(re_preprocess_text)
    return train

def preprocess(data_path):
    train = pd.read_csv(data_path)
    train = preprocess_dataframe(train)
    return train