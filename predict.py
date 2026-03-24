import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1] + 1

def predict_next_words(text, top_k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predictions = model.predict(token_list, verbose=0)[0]

    top_indices = predictions.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                results.append(word)
                break

    return results


# Test
print(predict_next_words("i love", top_k=5))
