import sqlite3

DB_NAME = "emotions.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        predicted_emotion TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_prediction(file_name, emotion, confidence):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (file_name, predicted_emotion, confidence)
        VALUES (?, ?, ?)
    """, (file_name, emotion, confidence))

    conn.commit()
    conn.close()

def get_predictions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows