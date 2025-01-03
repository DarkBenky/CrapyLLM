import csv
import random
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, LSTM, MultiHeadAttention, LayerNormalization,
    Dropout, Embedding, Input
)
import os

# Set the number of threads
num_threads = os.cpu_count()  # Use all available cores

# Configure TensorFlow to use multiple threads
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


# load whole text
def load_dataset():
    text = ''
    df = pd.read_csv('conversations.csv')
    for index, row in df.iterrows():
        text += row['Prompts'] + ' ' + row['Responses'] + ' '
    
    # load the text file 
    with open('text.txt', 'r') as file:
        text += file.read()
        
    return text

def simpleTokenizer(text, max_vocab_size = 10_000):
    text = text.split()
    word_count = {}
    for word in text:
        word = word.lower()
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
    vocab = list(word_count.keys())[:max_vocab_size]
    word2idx = {}
    idx2word = {}
    for i, word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word
    return word2idx, idx2word

MAX_SEQ_LEN = 512
BATCH_SIZE = 32
GENERATIONS = 1000

def create_model(vocab_size, seq_len, embedding_dim=256):
    inputs = Input(shape=(seq_len,))
    
    # Embedding
    x = Embedding(vocab_size, embedding_dim)(inputs)
    
    # Multiple LSTM + Multi-head Attention blocks, as before.
    for _ in range(6):
        lstm_out = LSTM(embedding_dim, return_sequences=True)(x)
        x = LayerNormalization()(lstm_out)

        attention = MultiHeadAttention(
            num_heads=8, 
            key_dim=embedding_dim // 8
        )(x, x, x)
        x = LayerNormalization()(x + attention)
        x = Dropout(0.1)(x)

    # Final LSTM with return_sequences=True so we get (batch, seq_len, embedding_dim)
    x = LSTM(embedding_dim, return_sequences=True)(x)
    x = Dropout(0.1)(x)


    # Dense layers
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)



    # Dense over each time step -> (batch, seq_len, vocab_size)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def generate_text(model, prompt, word2idx, idx2word, max_seq_len=512, num_tokens=64, temperature=1.0):
    """
    Generate up to num_tokens predictions given a user prompt.
    """
    # Convert prompt to token IDs
    tokens = []
    for word in prompt.split():
        tokens.append(word2idx.get(word, word2idx['<UNK>']))
    
    for _ in range(num_tokens):
        # Pad current tokens to max_seq_len
        padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            [tokens], maxlen=max_seq_len, padding='post'
        )
        
        # Predict next token for the last position
        # Model output shape: (1, max_seq_len, vocab_size)
        predictions = model.predict(padded_tokens, verbose=0)
        # Take the predictions from the last time-step
        last_step_preds = predictions[:, len(tokens) - 1, :] if len(tokens) < max_seq_len \
                          else predictions[:, -1, :]
        
        # Greedy pick the most likely next token
        # next_token_id = np.argmax(last_step_preds[0])

        # Sample from the distribution
        last_step_preds = np.log(last_step_preds) / temperature
        last_step_preds = np.exp(last_step_preds) / np.sum(np.exp(last_step_preds))
        next_token_id = np.random.choice(len(last_step_preds[0]), p=last_step_preds[0])
        
        # # Stop if next token is the padding index or out of vocab
        # if next_token_id == word2idx['<UNK>']:
        #     break
        
        # Append next token
        tokens.append(next_token_id)
        
        # Optional stopping if we reach max_seq_len
        if len(tokens) >= max_seq_len:
            break
    
    # Convert full tokens list back to words
    generated_words = []
    for t in tokens:
        generated_words.append(idx2word.get(t, '<UNK>'))
    return ' '.join(generated_words)

def train(model=None):
    text = load_dataset()
    word2idx, idx2word = simpleTokenizer(text, 16_000)
    print(f"Vocabulary size: {len(word2idx)}")
    word2idx['<UNK>'] = len(word2idx)
    vocab_size = len(word2idx)

    if model is None:
        model = create_model(vocab_size, MAX_SEQ_LEN, embedding_dim=1024)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    model.summary()

    best_accuracy = 0.0
    best_loss = 100.0

    for i in range(GENERATIONS):
        X = []
        Y = []
        for j in range(BATCH_SIZE):
            start = random.randint(0, len(text) - MAX_SEQ_LEN)
            end = start + MAX_SEQ_LEN
            seq = text[start:end]
            seq_indices = []
            for word in seq:
                seq_indices.append(word2idx.get(word, word2idx['<UNK>']))
            # X is the sequence up to the second-to-last word
            X.append(seq_indices[:-1])
            Y.append(seq_indices[1:])
        
        X = tf.keras.preprocessing.sequence.pad_sequences(
            X, maxlen=MAX_SEQ_LEN, padding='post'
        )
        Y = tf.keras.preprocessing.sequence.pad_sequences(
            Y, maxlen=MAX_SEQ_LEN, padding='post'
        )
        
        print(f"Generation {i + 1}/{GENERATIONS + 1}")
        model.fit(X, Y, batch_size=BATCH_SIZE, epochs=10, verbose=1)

        # print(model.history.history['accuracy'][-1])
        # if best_accuracy < model.history.history['accuracy'][-1]:
        #     best_accuracy = model.history.history['accuracy'][-1]
        #     model.save('simple_model.keras')

        print(model.history.history['loss'][-1])
        if best_loss > model.history.history['loss'][-1]:
            best_loss = model.history.history['loss'][-1]
            model.save('simple_model.keras')
    
    # model.save('simple_model.keras')

    # -----------------------------------------------
    # After training, interactive CLI for generation
    # -----------------------------------------------
    while True:
        user_input = input("\nEnter your prompt (or 'exit' to quit): ")
        if user_input.lower().strip() in ["exit", "quit"]:
            break
        
        user_input = user_input.lower()

        generated_text = generate_text(
            model=model,
            prompt=user_input,
            word2idx=word2idx,
            idx2word=idx2word,
            max_seq_len=MAX_SEQ_LEN,
            num_tokens=128,
            temperature=1.0
        )
        print(f"\nGenerated: {generated_text}")

    return model, word2idx, idx2word


# load pre-trained model
# model = tf.keras.models.load_model('simple_model.keras')
train()

