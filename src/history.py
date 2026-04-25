# src/history.py
# Prediction history system — stores and retrieves past predictions using SQLite.

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.db")


def _get_connection():
    """Get a connection to the SQLite database, creating the table if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path    TEXT    NOT NULL,
            predicted_label TEXT  NOT NULL,
            confidence    REAL   NOT NULL,
            timestamp     TEXT   NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_prediction(image_path, predicted_label, confidence):
    """
    Save a prediction record to the database.

    Args:
        image_path (str): Path to the input image.
        predicted_label (str): The predicted Pokémon name.
        confidence (float): Prediction confidence (0–1).
    """
    conn = _get_connection()
    conn.execute(
        "INSERT INTO predictions (image_path, predicted_label, confidence, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (image_path, predicted_label, confidence, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_history(limit=50):
    """
    Retrieve the most recent prediction records.

    Args:
        limit (int): Maximum number of records to return.

    Returns:
        list[dict]: Each dict contains id, image_path, predicted_label,
                    confidence, and timestamp.
    """
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT id, image_path, predicted_label, confidence, timestamp "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "image_path": r[1],
            "predicted_label": r[2],
            "confidence": r[3],
            "timestamp": r[4],
        }
        for r in rows
    ]


def clear_history():
    """Delete all prediction records."""
    conn = _get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Quick test
    log_prediction("test/pikachu.png", "pikachu", 0.95)
    print(get_history())
