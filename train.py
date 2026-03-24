import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical

# =========================
# 1. LOAD DATA FROM FILE
# =========================
def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if len(line.split()) > 2:  # ignore very short lines
                data.append(line)
    return data

# =========================
# 2. CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load and clean dataset
data = load_data("text.txt")
data = [clean_text(line) for line in data]

print("Total sentences:", len(data))
print("Sample data:", data[:5])

# =========================
# 3. TOKENIZATION
# =========================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

total_words = len(tokenizer.word_index) + 1
print("Vocabulary size:", total_words)

# =========================
# 4. CREATE SEQUENCES
# =========================
input_sequences = []

for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(token_list)):
        input_sequences.append(token_list[:i+1])

print("Total sequences:", len(input_sequences))

# =========================
# 5. PAD SEQUENCES
# =========================
max_len = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode labels
y = to_categorical(y, num_classes=total_words)

# =========================
# 6. BUILD MODEL (IMPROVED)
# =========================
model = Sequential()
model.add(Embedding(total_words, 200, input_length=max_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# =========================
# 7. TRAIN MODEL
# =========================
model.fit(X, y, epochs=30, batch_size=128, verbose=1)

# =========================
# 8. SAVE MODEL
# =========================
model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Training complete and model saved!")