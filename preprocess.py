import os

# Fix for Mac Apple Silicon thread deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# EMO-DB Emotion Mapping (German to English)
EMOTION_MAP = {
    'W': 'Angry', 'L': 'Boredom', 'E': 'Disgust', 'A': 'Fear',
    'F': 'Happy', 'T': 'Sad', 'N': 'Neutral'
}

TARGET_SR = 22050
MAX_DURATION = 3.0  # seconds
MAX_LEN = int(TARGET_SR * MAX_DURATION)


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=MAX_DURATION)

        # 1. Remove silence and normalize
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)

        # Pad or truncate to fixed length
        if len(y) > MAX_LEN:
            y = y[:MAX_LEN]
        else:
            y = np.pad(y, (0, MAX_LEN - len(y)), 'constant')

        # 2. Extract Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        # Convert to log scale for better representation
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return {
            'mfcc': mfccs,
            'mel_spect': mel_spect,
            'chroma': chroma,
            'contrast': contrast,
            'zcr': zcr,
            'rms': rms,
            'raw_audio': y
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def prepare_dataset(data_dir, save_dir='processed_data'):
    os.makedirs(save_dir, exist_ok=True)
    X_mfcc, X_mel, y_labels = [], [], []

    print("Extracting features from EMO-DB...")
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith('.wav'):
            # EMO-DB filename format: e.g., 03a01Fa.wav (6th char is emotion)
            emotion_code = file[5]
            if emotion_code in EMOTION_MAP:
                file_path = os.path.join(data_dir, file)
                features = extract_features(file_path)
                if features:
                    X_mfcc.append(features['mfcc'].T)      # Shape: (time, 40) for RNN/LSTM
                    X_mel.append(features['mel_spect'].T)  # Shape: (time, 128) for CNN
                    y_labels.append(EMOTION_MAP[emotion_code])

    # Convert to numpy arrays
    X_mfcc = np.array(X_mfcc)
    X_mel = np.array(X_mel)
    y_labels = np.array(y_labels)

    # Save raw audio for UI visualization
    np.save(os.path.join(save_dir, 'X_mfcc.npy'), X_mfcc)
    np.save(os.path.join(save_dir, 'X_mel.npy'), X_mel)
    np.save(os.path.join(save_dir, 'y_labels.npy'), y_labels)
    print(f"Dataset saved to {save_dir}")


if __name__ == "__main__":
    # Change this to your actual EMO-DB path
    EMO_DB_PATH = '/Users/dvs/Downloads/wav'
    prepare_dataset(EMO_DB_PATH)