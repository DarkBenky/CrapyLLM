import os
import sys

# Ensure compatibility with Keras and TensorFlow
os.environ['TF_KERAS'] = '1'

import pandas as pd
import numpy as np
import tensorflow as tf
import tf_keras  # Explicitly import tf-keras

from transformers import AutoTokenizer, TFAutoModelForCausalLM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description='Train or use a text generation model')
parser.add_argument('--mode', choices=['train', 'generate'], default='generate', 
                    help='Mode of operation: train the model or generate text')
parser.add_argument('--model_path', type=str, default='local_text_generation_model', 
                    help='Path to save or load the model')
parser.add_argument('--temperature', type=float, default=0.7, 
                    help='Sampling temperature for text generation')
args = parser.parse_args()

# Compatibility Check and Setup
def check_tensorflow_keras_compatibility():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    if not hasattr(tf.keras, '_tfkeras'):
        print("Warning: Using standard Keras. Some compatibility issues may occur.")
        print("Recommended: Install tf-keras with 'pip install tf-keras'")

check_tensorflow_keras_compatibility()

# Load Dataset
print("Loading dataset...")
df = pd.read_csv('conversations.csv')

# Preprocessing
def preprocess_text(row):
    prompt = row.get("Prompts", "")
    response = row.get("Responses", "")
    return f"{prompt} {response}".strip()

# Prepare Dataset
texts = df.apply(preprocess_text, axis=1).tolist()

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize Dataset
def tokenize_texts(texts):
    return tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )

# Training Mode
if args.mode == 'train':
    # Prepare Tokenized Dataset
    tokenized_dataset = tokenize_texts(texts)

    # Use local save path instead of Hugging Face repo
    local_model_path = args.model_path

    # Initialize Model
    model = TFAutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    # Prepare Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(
        filepath=local_model_path,
        save_best_only=True,
        save_weights_only=False
    )

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    # Prepare TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': tokenized_dataset['input_ids'],
        'attention_mask': tokenized_dataset['attention_mask'],
        'labels': tokenized_dataset['input_ids']
    }).shuffle(buffer_size=1000).batch(8)

    # Train Model
    model.fit(
        dataset, 
        epochs=3, 
        callbacks=[tensorboard_callback, checkpoint_callback]
    )

    # Save model and tokenizer locally
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

# Generation Mode
elif args.mode == 'generate':
    try:
        # Attempt to load local saved model
        local_model_path = args.model_path
        model = TFAutoModelForCausalLM.from_pretrained(local_model_path)
    except:
        # Fallback to default pre-trained model
        print("No local model found. Using pre-trained GPT-2 model.")
        model = TFAutoModelForCausalLM.from_pretrained("gpt2")

    # Interactive Generation Loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Tokenize input
        input_ids = tokenizer.encode(user_input, return_tensors='tf')
        
        # Generate response
        output = model.generate(
            input_ids, 
            max_length=100, 
            num_return_sequences=1, 
            temperature=args.temperature,
            no_repeat_ngram_size=2
        )
        
        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Model: {response}")

    print("Conversation ended.")

if __name__ == '__main__':
    print("Text Generation Model Ready.")