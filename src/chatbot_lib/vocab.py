import pandas as pd
import os
import pickle

from chatbot_lib.consts import REFERENCES_PATH

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Vocabulary:
    def __init__(self):
        self.trimmed = False
        self.reset_vocab()

    def reset_vocab(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def trim(self, min_count=1):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for word, count in self.word2count.items():
            if count >= min_count:
                keep_words.append(word)

        self.reset_vocab()
        for word in keep_words:
            self.add_word(word)

    def __len__(self):
        return self.num_words
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.word2index.get(item, None)
        elif isinstance(item, int):
            return self.index2word.get(item, None)
        else:
            raise TypeError("Item must be either a string or an integer.")
        
    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.word2index
        elif isinstance(item, int):
            return item in self.index2word
        else:
            raise TypeError("Item must be either a string or an integer.")
        
def add_text_to_vocab(text: str, vocab: Vocabulary) -> None:
    """
    Adds the text to the vocabulary.
    """
    vocab.add_sentence(text)

def trim_vocab(conv_df: pd.DataFrame, vocab: Vocabulary, min_count: int = 1) -> pd.DataFrame:
    """
    Trims the vocabulary to only include words that appear at least 'min_count' times.
    """
    vocab.trim(min_count)

    # Now update the DataFrame to remove words not in the trimmed vocabulary
    def filter_text(text):
        return ' '.join([word for word in text.split() if word in vocab])
    
    conv_df_trimmed = conv_df.copy()
    conv_df_trimmed["text1"] = conv_df_trimmed["text1"].apply(filter_text)
    conv_df_trimmed["text2"] = conv_df_trimmed["text2"].apply(filter_text)

    conv_df_trimmed = conv_df_trimmed[(conv_df_trimmed["text1"].str.strip() != "") & (conv_df_trimmed["text2"].str.strip() != "")]

    return conv_df_trimmed

def vectorize_text(text: str, vocab: Vocabulary) -> list:
    """
    Converts a text to a vector of indices based on the vocabulary.
    """
    return [vocab[word] for word in text.split() if word in vocab]

def save_vocab(vocab: Vocabulary) -> None:
    """
    Saves the vocabulary to a file.
    """
    os.makedirs(REFERENCES_PATH, exist_ok=True)  # Ensure the directory exists
    with open(f"{REFERENCES_PATH}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

def load_vocab() -> Vocabulary:
    with open(f"{REFERENCES_PATH}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        return vocab
    raise Exception(f"Vocabulary file ({REFERENCES_PATH}/vocab.pkl) not found")