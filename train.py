import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model import build_cnn_lstm

print("Loading preprocessed data...")
X_mfcc = np.load('processed_data/X_mfcc.npy')
y_labels = np.load('processed_data/y_labels.npy')

# Encode string labels to integers, then to one-hot vectors
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
y_categorical = to_categorical(y_encoded)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_mfcc, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

print(f"Training data shape: {X_train.shape}")
input_shape = X_train.shape[1:] # (time_steps, 40)

# Build, compile, and train the model
model = build_cnn_lstm(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=25, 
    batch_size=16
)

# Save the model for the Streamlit app
model.save('best_cnn-lstm_model.keras')
print("✅ Model saved as 'best_cnn-lstm_model.keras'")