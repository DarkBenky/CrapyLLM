import pandas as pd
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # Added import
import datetime
import tensorboard
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Train or load the model.')
parser.add_argument('--load_model', action='store_true', help='Load the model instead of training from scratch.')
parser.add_argument('--continue_training', action='store_true', help='Continue training the loaded model.')
parser.add_argument('--model_path', type=str, default='crappy_chatbot.h5', help='Path to the saved model.')
args = parser.parse_args()

# Initialize TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Initialize ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.keras',
    save_weights_only=False,
    save_freq='epoch'  # Saves the model after each epoch
)

# Load Dataset
print("Loading dataset...")
df = pd.read_csv('conversations.csv')

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nampdn-ai/tiny-codes")
# r = ds['train']['response']
# p = ds['train']['prompt']

# # add the new data to the existing dataframe
# df = df._append(pd.DataFrame({'Prompts': p, 'Responses': r}), ignore_index=True)

# Preprocessing Function
def preprocess_text(row):
    prompt = row.get("Prompts", "")
    response = row.get("Responses", "")
    # Combine prompt and response
    text = f"{prompt} {response}"
    return text.strip()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Model Parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 1024
lstm_units = 256
num_heads = 16
ff_dim = 1024
dropout_rate = 0.2
sequence_length = 4096  # Reduced sequence length for lower RAM usage
batch_size = 4

# Generator function
def data_generator():
    for _, row in df.iterrows():
        text = preprocess_text(row)
        encoding = tokenizer(
            text,
            max_length=sequence_length,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        label = tf.roll(input_ids, shift=-1, axis=0)
        yield {"inputs": input_ids, "attention_masks": attention_mask}, label

# Create Dataset
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=({"inputs": tf.int32, "attention_masks": tf.int32}, tf.int32),
    output_shapes=(
        {"inputs": (sequence_length,), "attention_masks": (sequence_length,)},
        (sequence_length,)
    )
)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Input Layers
inputs_input = Input(shape=(sequence_length,), dtype=tf.int32, name='inputs')
attention_masks_input = Input(shape=(sequence_length,), dtype=tf.int32, name='attention_masks')

if args.load_model:
    # Load the model from file
    model = tf.keras.models.load_model(args.model_path)
else:
    # Embedding Layer
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs_input)

    # Bidirectional LSTM Layer
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding)

    # Expand attention masks using Lambda layer
    expand_attention_masks = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(attention_masks_input)

    # Self-Attention Layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(
        query=x, value=x, attention_mask=expand_attention_masks
    )

    # Residual Connection and Layer Normalization
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    # Fully Connected Feedforward Layer
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output Layer
    outputs = Dense(vocab_size, activation='softmax')(x)

    # Compile Model
    model = Model(inputs=[inputs_input, attention_masks_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Summary
    model.summary()

# Training
if not args.load_model or args.continue_training:
    # Training with TensorBoard and ModelCheckpoint callbacks
    history = model.fit(
        dataset,
        epochs=3,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )

    # Save the final model
    model.save(args.model_path)
else:
    print('Model loaded and ready for use.')

def generate_text(model, initial_text, tokenizer, max_length=50, temperature=0.7):
    """
    Generate text sequentially using the model's predictions
    """
    # Initial tokenization
    context = tokenizer(
        initial_text,
        max_length=sequence_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    generated_tokens = context['input_ids'][0].numpy().tolist()
    generated_text = initial_text

    for _ in range(max_length):
        # Prepare input for model
        current_tokens = generated_tokens[-sequence_length:]
        current_input = tf.constant([current_tokens])
        current_mask = tf.ones_like(current_input)

        # Get model predictions
        predictions = model.predict(
            {'inputs': current_input, 'attention_masks': current_mask},
            verbose=0
        )

        # Get logits for the next token
        next_token_logits = predictions[0, -1, :] / temperature

        # Apply softmax for probabilities
        probs = tf.nn.softmax(next_token_logits).numpy()

        # Sample next token
        next_token = np.random.choice(len(probs), p=probs)

        # Append to generated sequence
        generated_tokens.append(next_token)

        # Decode new token
        new_text = tokenizer.decode([next_token], skip_special_tokens=True)
        generated_text += new_text

        # Check stopping conditions
        if new_text.strip() in ['.', '!', '?', '\n'] and len(generated_text.split()) > 5:
            break
        if '[SEP]' in new_text or '[PAD]' in new_text:
            break

    return generated_text

# Updated interactive loop
print("Model is ready. You can now interact with the model.")

while True:
    user_input = input("You: ")
    if 't=' in user_input:
        temperature = float(user_input.split('=')[1])
        print(f"Temperature set to {temperature}")
        continue
    if user_input.lower() in ['exit', 'quit']:
        break

    # Generate response
    response = generate_text(
        model=model,
        initial_text=user_input,
        tokenizer=tokenizer,
        max_length=50,
        temperature=0.1
    )

    print(f"Model: {response}")

print("Conversation ended.")