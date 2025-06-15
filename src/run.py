import torch

from chatbot_lib.model import Transformer, load_transformer
from chatbot_lib.vocab import Vocabulary, load_vocab
from chatbot_lib.utils import normalize_text, get_device

from chatbot_lib.consts import (
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    D_FF,
    MAX_SEQ_LENGTH,
    DROPOUT,
)

device = get_device()
vocab: Vocabulary = load_vocab()
transformer: Transformer = load_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT, device)

# Define a function to predict the response for a given input sequence
def predict_response(input_seq):
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    output = transformer(input_tensor, input_tensor[:, :-1])
    output_seq = output.argmax(dim=-1).squeeze().tolist()
    return output_seq if isinstance(output_seq, list) else [output_seq]

# Define a function to convert input sentence string to output sentence string
def input_to_output(input_sentence):
    input_sentence = normalize_text(input_sentence)
    if not input_sentence:
        return "Input sentence is empty."
    input_seq = [vocab[word] for word in input_sentence.split() if word in vocab]
    output_seq = predict_response(input_seq)
    output_sentence = ' '.join([vocab[index] for index in output_seq if index in vocab])
    return output_sentence

def main():
    print("Chat with the model! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        response = input_to_output(user_input)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()