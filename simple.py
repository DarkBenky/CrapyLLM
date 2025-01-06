import csv
import random
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, LSTM, MultiHeadAttention, LayerNormalization,
    Dropout, Embedding, Input, TextVectorization
)
import os
import json

# Constants
TOKENIZER_PATH = "tokenizer"
VOCAB_PATH = "vocabulary.json"
MAX_VOCAB_SIZE = 10000

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

def create_tokenizer(text, max_vocab_size=MAX_VOCAB_SIZE):
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = tf.keras.models.load_model(TOKENIZER_PATH)
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        return tokenizer, vocabulary
    
    # Create and configure tokenizer
    tokenizer = TextVectorization(
        max_tokens=max_vocab_size,
        standardize='lower_and_strip_punctuation',
        output_sequence_length=None,
        output_mode='int'
    )
    
    # Adapt tokenizer to the text
    tokenizer.adapt([text])
    
    # Save vocabulary
    vocabulary = dict(enumerate(tokenizer.get_vocabulary()))
    with open(VOCAB_PATH, 'w') as f:
        json.dump(vocabulary, f)
    
    # Save tokenizer
    model = tf.keras.Sequential([tokenizer])
    model.save(TOKENIZER_PATH)
    
    return tokenizer, vocabulary

def encode_text(tokenizer, text):
    return tokenizer(tf.constant([text]))[0]

def decode_text(vocabulary, sequence):
    return ' '.join([vocabulary.get(str(i), '?') for i in sequence.numpy()])

MAX_SEQ_LEN = 512
BATCH_SIZE = 32
GENERATIONS = 1

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

def generate_text(model, prompt, tokenizer, vocabulary, max_seq_len=512, num_tokens=64, temperature=1.0):
    """
    Generate text using TextVectorization tokenizer
    """
    # Convert prompt to token IDs
    current_tokens = tokenizer(tf.constant([prompt]))[0]
    tokens = current_tokens.numpy().tolist()
    
    for _ in range(num_tokens):
        # Pad sequence if needed
        if len(tokens) < max_seq_len:
            padded_tokens = tf.pad(
                [tokens], 
                [[0, 0], [0, max_seq_len - len(tokens)]], 
                'constant'
            )
        else:
            padded_tokens = [tokens[-max_seq_len:]]
            
        # Get model predictions
        predictions = model.predict(padded_tokens, verbose=0)
        last_step_preds = predictions[0, len(tokens)-1 if len(tokens) < max_seq_len else -1, :]
        
        if temperature == 0:
            # Greedy sampling
            next_token_id = np.argmax(last_step_preds)
        else:
            # Temperature sampling
            scaled_preds = last_step_preds / temperature
            scaled_preds = np.exp(scaled_preds) / np.sum(np.exp(scaled_preds))
            next_token_id = np.random.choice(len(scaled_preds), p=scaled_preds)
            
        # Skip special tokens and single characters
        vocab_word = vocabulary.get(str(next_token_id), '')
        if len(vocab_word) <= 1:
            # Get next best prediction
            sorted_indices = np.argsort(last_step_preds)[::-1]
            for idx in sorted_indices[1:]:
                if len(vocabulary.get(str(idx), '')) > 1:
                    next_token_id = idx
                    break
        
        tokens.append(next_token_id)
        
        # Stop if end token is generated
        if vocab_word == '</s>':
            break
            
    # Convert tokens back to text
    return decode_text(vocabulary, tf.constant(tokens))

def train(model=None):
    text = load_dataset()
    tokenizer, vocabulary = create_tokenizer(text, MAX_VOCAB_SIZE)
    word2idx = {word: idx for idx, word in vocabulary.items()}
    idx2word = {idx: word for word, idx in word2idx.items()}
    print(f"Vocabulary size: {len(word2idx)}")
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
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            max_seq_len=MAX_SEQ_LEN,
            num_tokens=128,
            temperature=1.0
        )
        print(f"\nGenerated: {generated_text}")

    return model, word2idx, idx2word


# load pre-trained model
model = tf.keras.models.load_model('simple_model.keras')
train(model)

