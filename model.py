import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Add these imports
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, LayerNormalization, Bidirectional, MultiHeadAttention, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import random

class TextProcessor:
    def __init__(self, max_vocab_size=10000, max_sequence_length=50):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(
            num_words=max_vocab_size, 
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.vocab_size = None

    def preprocess_text(self, df, tokenizer_path='tokenizer.json'):
        X = []  # Text inputs
        Y = []  # Next words

        COUNT = 128

        prompts = df.get("Prompts", "").tolist()
        responses = df.get("Responses", "").tolist()

        func_df = pd.read_csv('functions.csv')
        func_prompts = func_df.get("Prompts", "").tolist()
        func_responses = func_df.get("Responses", "").tolist()

        c = 0
        while c < COUNT // 1.5:
            random_index = random.randint(0, len(func_prompts) - 1)
            X.append(func_prompts.pop(random_index))
            Y.append(func_responses.pop(random_index))
            c += 1


        random_prompts = []
        random_responses = []

        while len(random_prompts) < COUNT * 1.5:
            random_index = random.randint(0, len(prompts) - 1)
            random_prompts.append(prompts.pop(random_index))
            random_responses.append(responses.pop(random_index))

        prompts = random_prompts
        responses = random_responses

        del random_prompts
        del random_responses

        for p, r in zip(prompts, responses):
            text = f"Prompt: {p} Response: "
            words = r.split()
            for i in range(1, len(words)):
                input_sequence = text + ' '.join(words[:i])
                next_word = words[i]
                X.append(input_sequence)
                Y.append(next_word)
                # input without prompt
                X.append('Response: '+' '.join(words[:i]))
                Y.append(next_word)

        # Train also on prompts
        for p in prompts:
            words = p.split()
            for i in range(1, len(words)):
                input_sequence = "Prompt: " + ' '.join(words[:i])
                next_word = words[i]
                X.append(input_sequence)
                Y.append(next_word)


        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            self.load_tokenizer(tokenizer_path)
        else:
            print("Creating new tokenizer...")

            self.tokenizer.fit_on_texts(X + Y)
            self.vocab_size = min(
                self.max_vocab_size, 
                len(self.tokenizer.word_index) + 1
            )

            self.save_tokenizer(tokenizer_path)


        X_seq = self.tokenizer.texts_to_sequences(X)
        Y_seq = self.tokenizer.texts_to_sequences(Y)


        X_padded = pad_sequences(
            X_seq, 
            maxlen=self.max_sequence_length,
            padding='pre',
            truncating='pre'
        )


        Y_processed = np.array([
            seq[0] if len(seq) > 0 else 0 for seq in Y_seq
        ])


        Y_encoded = tf.keras.utils.to_categorical(
            Y_processed, 
            num_classes=self.vocab_size
        )

        return np.array(X_padded), np.array(Y_encoded)

    def create_tf_dataset(self, X, Y, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def decode_sequence(self, sequence):
        return self.tokenizer.sequences_to_texts([sequence])[0]

    def save_tokenizer(self, path='tokenizer.json'):
        tokenizer_json = self.tokenizer.to_json()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)

    def load_tokenizer(self, path='tokenizer.json'):
        
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            self.vocab_size = min(
                self.max_vocab_size, 
                len(self.tokenizer.word_index) + 1
            )

class NextWordPredictor:
    def __init__(self, vocab_size, sequence_length, embedding_dim=128):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

    def build_model(self):
        # Input layer
        inputs = Input(shape=(self.sequence_length,))
        
        # Smaller embedding
        x = Embedding(self.vocab_size, 512)(inputs)
        x = LayerNormalization()(x)
        
        # Single Bidirectional LSTM with smaller units
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        
        # Simplified attention
        attention_output = MultiHeadAttention(
            num_heads=16, 
            key_dim=128
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Global average pooling instead of another LSTM
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(self.vocab_size, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model

    def train(self, train_dataset, epochs=10, checkpoint_path='next_word_model-bigger.keras'):
        checkpoint = ModelCheckpoint(
            checkpoint_path, 
            monitor='loss', 
            save_best_only=True
        )
        early_stop = EarlyStopping(monitor='loss', patience=3)
        self.model.fit(
            train_dataset, 
            epochs=epochs, 
            callbacks=[checkpoint, early_stop]
        )
        self.model.save(checkpoint_path)

    def load(self, model_path='next_word_model-bigger.keras'):
        self.model = load_model(model_path)

    def predict_next_word(self, input_text, tokenizer, temperature=1.0):
        input_seq = tokenizer.texts_to_sequences([input_text.lower()])
        input_seq = pad_sequences(
            input_seq, 
            maxlen=self.sequence_length, 
            padding='pre', 
            truncating='pre'
        )
        preds = self.model.predict(input_seq)[0]

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        probas = np.random.multinomial(1, preds, 1)
        predicted_index = np.argmax(probas)

        predicted_word = tokenizer.index_word.get(predicted_index, '')
        return predicted_word

if __name__ == "__main__":
    processor = TextProcessor(
        max_vocab_size=10000, 
        max_sequence_length=2048
    )
    df = pd.read_csv('conversations.csv')
    X_processed, Y_processed = processor.preprocess_text(df, 'tokenizer.json')
    train_dataset = processor.create_tf_dataset(X_processed, Y_processed, batch_size=16)
    print(f"Vocabulary size: {processor.vocab_size}")
    print(f"Input shape: {X_processed.shape}")
    print(f"Output shape: {Y_processed.shape}")

    if os.path.exists('next_word_model-bigger.keras'):
        print("Loading existing model...")
        predictor = NextWordPredictor(
            processor.vocab_size, 
            processor.max_sequence_length
        )
        predictor.load('next_word_model-bigger.keras')
        predictor.train(
            train_dataset, 
            epochs=100, 
            checkpoint_path='next_word_model-bigger.keras'
        )
    else:
        print("Training new model...")
        predictor = NextWordPredictor(
            processor.vocab_size, 
            processor.max_sequence_length
        )
        predictor.train(
            train_dataset, 
            epochs=100, 
            checkpoint_path='next_word_model-bigger.keras'
        )


    while True:
        input_text = input("Enter prompt (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
            
        input_text = "Prompt: " + input_text + " Response: "
        for _ in range(32):
            next_word = predictor.predict_next_word(
                input_text, 
                processor.tokenizer,
                temperature=0.25
            )
            input_text += next_word + " "
        

        print(f"Next word prediction: {input_text}")