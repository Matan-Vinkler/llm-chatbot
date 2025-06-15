import os
import pandas as pd

from chatbot_lib.utils import normalize_text, lemmatize_text, pad_sequence
from chatbot_lib.vocab import Vocabulary, add_text_to_vocab, trim_vocab, vectorize_text, save_vocab, PAD_TOKEN

from chatbot_lib.consts import DATA_RAW_PATH, DATA_INTERIM_PATH, DATA_PROCESSED_PATH

def load_conversations_and_lines() -> pd.DataFrame:
    conversations_df = pd.read_csv(os.path.join(DATA_RAW_PATH, 'movie_conversations.tsv'), sep='\t', header=None, on_bad_lines='skip')
    conversations_df.columns = ["char_id1", "char_id2", "movie_id", "frase_ids"]

    lines_df = pd.read_csv(os.path.join(DATA_RAW_PATH, 'movie_lines.tsv'), sep='\t', header=None, on_bad_lines='skip')
    lines_df.columns = ["line_id", "char_id", "movie_id", "char_name", "text"]

    return conversations_df, lines_df

def split_frase_ids(df):
    new_rows = []
    for _, row in df.iterrows():
        # Clean and split the frase_ids string
        frase_ids = row['frase_ids'].replace("'", "").replace("[", "").replace("]", "").split()
        if len(frase_ids) >= 2:
            for i in range(len(frase_ids) - 1):
                row_copy = row.copy()
                row_copy['frase_ids'] = str([frase_ids[i], frase_ids[i+1]])
                new_rows.append(row_copy)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows)

def preprocess_conversations(conversations_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    conversations_df = split_frase_ids(conversations_df)

    conversations_df[['frase_id1', 'frase_id2']] = conversations_df['frase_ids'].apply(lambda x: pd.Series(eval(x)))
    conversations_df.drop(columns=['frase_ids'], inplace=True)

    merged_df = conversations_df.merge(
        lines_df[['line_id', 'text']].rename(columns={'line_id': 'frase_id1', 'text': 'text1'}),
        on='frase_id1', how='left'
    ).merge(
        lines_df[['line_id', 'text']].rename(columns={'line_id': 'frase_id2', 'text': 'text2'}),
        on='frase_id2', how='left'
    )

    merged_df = merged_df[['text1','text2']]
    merged_df.dropna(inplace=True)

    merged_df["text1"] = merged_df["text1"].apply(normalize_text)
    merged_df["text2"] = merged_df["text2"].apply(normalize_text)

    return merged_df

def load_and_preprocess_data() -> pd.DataFrame:
    conversations_df, lines_df = load_conversations_and_lines()
    preprocessed_df = preprocess_conversations(conversations_df, lines_df)
    
    # Save the preprocessed DataFrame to a CSV file
    preprocessed_df.to_csv(os.path.join(DATA_INTERIM_PATH, 'preprocessed_conversations.csv'), index=False)
    
    return preprocessed_df

def load_preprocessed_data() -> pd.DataFrame:
    """
    Loads the preprocessed DataFrame from the interim directory.
    """
    preprocessed_file_path = os.path.join(DATA_INTERIM_PATH, 'preprocessed_conversations.csv')
    if not os.path.exists(preprocessed_file_path):
        raise FileNotFoundError(f"Preprocessed data file not found: {preprocessed_file_path}")
    
    df = pd.read_csv(preprocessed_file_path)
    df.dropna(inplace=True)
    return df

def vectorize_preprocessed_data(df: pd.DataFrame, vocab: Vocabulary) -> pd.DataFrame:
    """
    Converts the preprocessed DataFrame to a vectorized format using the provided vocabulary.
    """

    df["text1"] = df["text1"].apply(lemmatize_text)
    df["text2"] = df["text2"].apply(lemmatize_text)

    df["text1"].apply(lambda x: add_text_to_vocab(x, vocab))
    df["text2"].apply(lambda x: add_text_to_vocab(x, vocab))

    MIN_COUNT = 3
    df = trim_vocab(df, vocab, min_count=MIN_COUNT)

    df["seq1"] = df["text1"].apply(lambda x: vectorize_text(x, vocab))
    df["seq2"] = df["text2"].apply(lambda x: vectorize_text(x, vocab))

    df.drop(columns=["text1", "text2"], inplace=True)

    max_length = max(df["seq1"].apply(len).max(), df["seq2"].apply(len).max())

    df["seq1"] = df["seq1"].apply(lambda x: pad_sequence(x, max_length, PAD_TOKEN))
    df["seq2"] = df["seq2"].apply(lambda x: pad_sequence(x, max_length, PAD_TOKEN))

    df_save = df.copy()

    df_save["seq1"] = df_save["seq1"].apply(lambda x: ','.join(map(str, x)))
    df_save["seq2"] = df_save["seq2"].apply(lambda x: ','.join(map(str, x)))

    df_save.to_csv(os.path.join(DATA_PROCESSED_PATH, 'conversations_vectorized.csv'), index=False)
    save_vocab(vocab)
    
    return df

def load_vectorized_data() -> pd.DataFrame:
    """
    Loads the vectorized DataFrame from the processed directory.
    """
    vectorized_file_path = os.path.join(DATA_PROCESSED_PATH, 'conversations_vectorized.csv')
    if not os.path.exists(vectorized_file_path):
        raise FileNotFoundError(f"Vectorized data file not found: {vectorized_file_path}")
    
    df = pd.read_csv(vectorized_file_path)
    
    # Convert string representations of lists back to actual lists
    df["seq1"] = df["seq1"].apply(lambda x: [int(i) for i in x.split(",") if i.strip().isdigit()])
    df["seq2"] = df["seq2"].apply(lambda x: [int(i) for i in x.split(",") if i.strip().isdigit()])
    
    return df