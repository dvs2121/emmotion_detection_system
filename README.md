# Emotion Recognition from Speech Using CNN, RNN, LSTM and MFCC Features

## Objective
Develop an intelligent speech emotion recognition system capable of identifying human emotions (Happy, Angry, Sad, Neutral, Fearful, Disgust, Boredom) from audio recordings using deep learning.

## Dataset
This project uses the **EMO-DB (Berlin Emotional Speech Database)**.

Download the dataset and place the extracted `.wav` files inside the `data/EMO-DB/` directory.

## Setup & Installation

1. Install dependencies:

    pip install -r requirements.txt

2. Train the model:

    python train.py

3. Run prediction:

    python predict.py --audio sample.wav

## Project Structure

Speech_Emotion_Recognition/
│
├── data/
├── models/
├── notebooks/
├── src/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
