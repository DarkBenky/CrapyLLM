import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, 
    MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Hyperparameters
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 512
EPOCHS = 10
BATCH_SIZE = 64
NUMBER_OF_LSTM_UNITS = 256
NUMBER_OF_LSTM_LAYERS = 2

# Load and Preprocess Dataset
def load_and_preprocess_data(file_path, dataset_External = False):
    df = pd.read_csv(file_path)
    
    # Combine prompts and responses
    texts = []

    if dataset_External:
        from datasets import load_dataset

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("nampdn-ai/tiny-codes")
        r = ds['train']['response']
        p = ds['train']['prompt']

        for rs , pr in zip(r, p):
            texts.append(pr + ' ' + rs)  
    
    for _, row in df.iterrows():
        prompt = str(row.get('Prompts', ''))
        response = str(row.get('Responses', ''))
        texts.append(prompt + ' ' + response)
    
    return texts

# Tokenization
def tokenize_texts(texts, max_words, max_sequence_length):
    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, 
                                     padding='post', truncating='post')
    
    # Prepare input and output sequences for next token prediction
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, 1:]
    
    return tokenizer, X, y

# Create Custom Text Generation Model
def create_text_model(vocab_size, max_sequence_length, embedding_dim):
    inputs = Input(shape=(max_sequence_length-1,))
    
    # Embedding Layer
    x = Embedding(vocab_size, embedding_dim, input_length=max_sequence_length-1)(inputs)
    
    # LSTM layers with proper sequence return
    for _ in range(NUMBER_OF_LSTM_LAYERS-1):
        x = LSTM(NUMBER_OF_LSTM_UNITS, return_sequences=True)(x)
        x = Dropout(0.3)(x)
    
     # Final LSTM layer
    x = LSTM(NUMBER_OF_LSTM_UNITS, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    # Multi-Head Attention
    x = MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(x, x)
    
    # Layer Normalization
    x = LayerNormalization()(x)
    
    # Output Layer
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# Text Generation Function
def generate_text(model, tokenizer, seed_text, max_sequence_length, next_words=50):
    for _ in range(next_words):
        # Tokenize and pad the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='post', truncating='post')
        
        temperature = 0.7

        # Predict next token
        predicted = model.predict(token_list, verbose=0)
        next_token_logits = predicted[0, -1, :] / temperature

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(next_token_logits).numpy()

        # Sample from the probability distribution
        predicted_word_index = np.random.choice(len(probabilities), p=probabilities)

        # Convert index to word
        output_word = tokenizer.index_word[predicted_word_index]
        
         # Append the word to the seed text
        seed_text += " " + output_word

        # Safety check for maximum length
        if len(seed_text.split()) > max_sequence_length:
            break
    
    return seed_text

# Main Training and Generation Script
def main(ext = False):
    # Load and preprocess data
    texts = load_and_preprocess_data('conversations.csv', ext)
    
    # Tokenize texts
    tokenizer, X, y = tokenize_texts(texts, MAX_WORDS, MAX_SEQUENCE_LENGTH)
    
    # Get vocab size
    vocab_size = len(tokenizer.word_index) + 1
    
    # Create model
    # Check if model already exists
    if os.path.exists('text_generation_model.keras'):
        print("Model already exists. Skipping training.")
        # load model
        model = load_model('text_generation_model.keras')
    else:
        model = create_text_model(vocab_size, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        'text_generation_model.keras', 
        monitor='loss', 
        save_best_only=True
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=[checkpoint, early_stop]
    )
    
    # Save tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model training completed and saved.")

# Interactive Generation
def interact_with_model():
    # Load model and tokenizer
    model = load_model('text_generation_model.keras')
    
    import pickle
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("Model loaded. Start chatting (type 'exit' to quit):")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Generate response
        generated_text = generate_text(
            model, 
            tokenizer, 
            user_input, 
            MAX_SEQUENCE_LENGTH
        )
        
        print(f"Model: {generated_text}")
    
    print("Conversation ended.")

if __name__ == '__main__':
    # Choose either training or interaction
    mode = input("Enter 'train' to train the model or 'chat' to interact: ").lower()
    
    if mode == 'train':
        main()
    elif mode == 'chat':
        interact_with_model()
    elif mode == 't':
        main(True)
    else:
        print("Invalid mode. Please enter 'train' or 'chat'.")