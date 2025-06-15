import torch
import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemma = WordNetLemmatizer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def lemmatize_text(text):
    """Lemmatize the input text."""
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = []
    for word, tag in nltk.pos_tag(tokens):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'n', 'v'] else None
        if wntag:
            lemmatized_tokens.append(lemma.lemmatize(word, pos=wntag))
        else:
            lemmatized_tokens.append(word)
    return ' '.join(lemmatized_tokens)

def pad_sequence(sequence, max_length, pad_value):
    """
    Pads a sequence to the specified max_length with the pad_value.
    """
    return sequence + [pad_value] * (max_length - len(sequence))

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_default_device(torch.device("cuda:0"))
        print("Using GPU")
    else:
        print("No GPU available, using CPU")

    torch.cuda.empty_cache()
    return device
