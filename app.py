from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json
import datetime
import sqlite3

# connect to the database
conn = sqlite3.connect('db.db', check_same_thread=False)  # Add check_same_thread=False for Flask

from model import TextProcessor, NextWordPredictor

app = Flask(__name__)

ModelError = False

try:
    # Initialize TextProcessor
    processor = TextProcessor(
        max_vocab_size=10000, 
        max_sequence_length=512
    )

    tokenizer_path = 'tokenizer.json'
    if os.path.exists(tokenizer_path):
        processor.load_tokenizer(tokenizer_path)
        print("Tokenizer loaded.")
    else:
        raise FileNotFoundError("Tokenizer file not found. Please train the model first.")

    model_path = 'next_word_model-big.keras'
    predictor = NextWordPredictor(
        vocab_size=processor.vocab_size, 
        sequence_length=processor.max_sequence_length
    )

    if os.path.exists(model_path):
        predictor.load(model_path)
        print("Model loaded.")
    else:
        raise FileNotFoundError("Model file not found. Please train the model first.")
except FileNotFoundError as e:
    print(e)
    ModelError = True


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        prompt = user_input
        user_input = "Prompt:" + user_input + " Response: "
        if user_input.lower() in ['exit', 'quit']:
            return render_template('index.html', prediction="Goodbye!")
        
        if ModelError:
            return render_template('index.html', prediction="Error loading model. Please train the model first.")
        
        response = ""
        temperate = 0.25
        for _ in range(128):
            next_word = predictor.predict_next_word(user_input, processor.tokenizer, temperature=temperate)
            if "<OOV>" in next_word:
                next_word = ""
                temperate += 0.01
            user_input += next_word + " "
            prediction = f"{user_input} {next_word}"
            response += next_word + " "

        current_time = datetime.datetime.now()

        # Insert the conversation into the database
        conn.execute(
            "INSERT INTO conversations (Prompts, Responses, Times) VALUES (?, ?, ?)",
            (prompt, response, current_time)
        )

        conn.commit()
        
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html', prediction="")

@app.route('/conversations')
def conversations():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total_conversations = cursor.fetchone()[0]

    cursor.execute("SELECT Prompts, Responses, Times FROM conversations ORDER BY Times DESC LIMIT ? OFFSET ?", (per_page, offset))
    conversations = cursor.fetchall()

    next_url = url_for('conversations', page=page + 1) if offset + per_page < total_conversations else None
    prev_url = url_for('conversations', page=page - 1) if page > 1 else None

    return render_template('conversations.html', conversations=conversations, next_url=next_url, prev_url=prev_url)

if __name__ == '__main__':
    app.run(debug=True)