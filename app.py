import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LSTM, Dense, 
                                      Dropout, Flatten, Reshape, Input)
from tensorflow.keras.callbacks import EarlyStopping
import gradio as gr
from datetime import datetime
import json

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = './emodb/wav'
MODEL_SAVE_PATH = 'emotion_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.npy'
HISTORY_FILE = 'prediction_history.json'

EMODB_LABELS = {
    'W': 'Anger',
    'L': 'Boredom',
    'E': 'Disgust',
    'A': 'Anxiety/Fear',
    'F': 'Happiness',
    'T': 'Sadness',
    'N': 'Neutral'
}

N_MFCC = 40
MAX_PAD_LEN = 150
TARGET_SR = 16000

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=TARGET_SR, duration=4.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ==========================================
# DATA LOADING  ← THIS WAS MISSING!
# ==========================================
def load_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"❌ Dataset not found at: {dataset_path}\n"
            f"Please download EMODB and place it in './emodb/wav/'\n"
            f"Download from: http://emodb.bund.de/en/download/"
        )
    
    features, labels = [], []
    wav_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"❌ No .wav files found in {dataset_path}")
    
    print(f"Found {len(wav_files)} audio files. Extracting features...")
    
    for filename in wav_files:
        emotion_char = filename[5]
        if emotion_char in EMODB_LABELS:
            mfccs = extract_features(os.path.join(dataset_path, filename))
            if mfccs is not None:
                features.append(mfccs)
                labels.append(EMODB_LABELS[emotion_char])
    
    print(f"✅ Loaded {len(features)} samples across {len(set(labels))} emotions")
    return np.array(features), np.array(labels)

# ==========================================
# MODEL BUILDING
# ==========================================
def build_crnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        
        # CNN Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # CNN Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Reshape for LSTM: (10, 37, 64) → (37, 640)
        Reshape((37, 10 * 64)),
        
        # LSTM
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        
        # Classifier
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# PREDICTION HISTORY
# ==========================================
def load_prediction_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_prediction_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_to_history(filename, prediction, confidence):
    history = load_prediction_history()
    history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': os.path.basename(filename),
        'prediction': prediction,
        'confidence': round(confidence, 2)
    })
    if len(history) > 20:
        history = history[-20:]
    save_prediction_history(history)

def get_history_display():
    history = load_prediction_history()
    if not history:
        return "📊 **Prediction History**: No predictions yet."
    
    text = "📊 **Prediction History** (Last 20)\n\n"
    text += "| Time | File | Emotion | Conf. |\n"
    text += "|------|------|---------|-------|\n"
    for entry in reversed(history):
        text += f"| {entry['timestamp']} | {entry['filename']} | {entry['prediction']} | {entry['confidence']}% |\n"
    return text

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_emotion(audio_path, model, label_encoder):
    if audio_path is None:
        return "❌ No audio file uploaded!", 0, None
    
    mfccs = extract_features(audio_path)
    if mfccs is None:
        return "❌ Error processing audio. Try a .wav file.", 0, None
    
    mfccs = mfccs[..., np.newaxis]
    mfccs = np.expand_dims(mfccs, axis=0)
    
    predictions = model.predict(mfccs, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(predictions) * 100
    
    prob_text = "**All Emotion Probabilities:**\n\n"
    for i, label in enumerate(label_encoder.classes_):
        prob = predictions[0][i] * 100
        bar = "█" * int(prob / 5)
        prob_text += f"{label:15}: {prob:5.2f}% {bar}\n"
    
    add_to_history(audio_path, predicted_emotion, confidence)
    
    result_text = f"""
## 🎙️ Emotion Detection Result

**Predicted Emotion:** {predicted_emotion}

**Confidence:** {confidence:.2f}%

---

{prob_text}
"""
    return result_text, confidence, predictions[0]

# ==========================================
# GRADIO INTERFACE
# ==========================================
def create_interface(model, label_encoder):
    def gradio_predict(audio_file):
        result_text, confidence, probs = predict_emotion(audio_file, model, label_encoder)
        history_text = get_history_display()
        return result_text, history_text
    
    def clear_history():
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return "✅ History cleared!", "📊 **Prediction History**: No predictions yet."
    
    with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 Speech Emotion Recognition System
        Upload an audio file to detect emotions.
        
        **Note:** Model trained on German speech (EMO-DB). Works best with short clips (2-4 seconds).
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(label="Upload Audio", type="filepath",
                                       sources=["upload", "microphone"])
                submit_btn = gr.Button("🎯 Detect Emotion", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            with gr.Column(scale=3):
                output_text = gr.Markdown(label="Result")
        
        with gr.Row():
            history_text = gr.Markdown(label="History")
        
        submit_btn.click(fn=gradio_predict, inputs=[audio_input],
                         outputs=[output_text, history_text])
        clear_btn.click(fn=clear_history, inputs=[],
                        outputs=[output_text, history_text])
        demo.load(fn=lambda: ("", get_history_display()),
                  outputs=[output_text, history_text])
    
    return demo

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPEECH EMOTION RECOGNITION SYSTEM")
    print("="*60)
    
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print(f"\n✅ Loading saved model...")
        model = load_model(MODEL_SAVE_PATH)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
        print("✅ Model loaded!")
    else:
        print("\n⚠️  No saved model found. Training from scratch...")
        
        # ✅ FIX: Pass DATASET_PATH as argument
        X, y = load_data("/Users/dvs/Downloads/wav")
        X = X[..., np.newaxis]
        
        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_int)
        num_classes = len(label_encoder.classes_)
        
        np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
        
        input_shape = (N_MFCC, MAX_PAD_LEN, 1)
        model = build_crnn_model(input_shape, num_classes)
        model.summary()
        
        print("\n🚀 Starting training...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                   restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            batch_size=16, epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stop]
        )
        
        model.save(MODEL_SAVE_PATH)
        print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    print("\n🌐 Launching Web Interface...")
    demo = create_interface(model, label_encoder)
    demo.launch(server_name="0.0.0.0", server_port=7860, 
                share=False, inbrowser=True)
hide_footer = """
footer {
    visibility: hidden !important;
}

.gradio-container footer {
    display: none !important;
}

#api-banner {
    display: none !important;
}
"""