import pandas as pd
import sqlite3

# Load data from CSV
df = pd.read_csv('conversations.csv')

# Connect to SQLite database
conn = sqlite3.connect('db.db')

# Crate teble in database for conversations Layout [Prompt, Response, Timestamp]

df.to_sql('conversations', conn, if_exists='replace', index=False)

# Commit changes
conn.commit()




