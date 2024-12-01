import pandas as pd
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention,
    LayerNormalization, Add
)

# Load Dataset
print("Loading dataset...")
df = pd.read_csv('conversations.csv')

# Preprocessing Function
def preprocess_text(row):
    prompt = row.get("Prompts", "")
    response = row.get("Responses", "")
    text = f"{prompt} {response}"
    return text.strip()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Model Parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 128
lstm_units = 256
num_heads = 4
ff_dim = 512
dropout_rate = 0.25
sequence_length = 512  # Reduced sequence length
batch_size = 32

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
history = model.fit(dataset, epochs=5)