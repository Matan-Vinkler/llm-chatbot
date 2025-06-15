import os

REFERENCES_PATH = os.path.join(os.getcwd(), "references")
MODELS_PATH = os.path.join(os.getcwd(), "models")

DATA_INTERIM_PATH = os.path.join(os.getcwd(), 'data/interim')
DATA_RAW_PATH = os.path.join(os.getcwd(), 'data/raw')
DATA_PROCESSED_PATH = os.path.join(os.getcwd(), 'data/processed')

SRC_VOCAB_SIZE = 23570
TGT_VOCAB_SIZE = 23570
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048
MAX_SEQ_LENGTH = 464
DROPOUT = 0.1

BATCH_SIZE = 16
NUM_EPOCHS = 100