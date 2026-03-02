import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model_utils import build_lstm_model

# 1. Exact Match Configuration
ACTIONS = np.array(['Hello', 'How are you', 'Sorry', 'I need help', 'Thank you'])
DATA_PATH = "LSTM_Dataset"
SEQ_COUNT = 30
FRAME_COUNT = 20 # The 0.8 second window

print("📂 Loading 45-Frame Sequences...")
label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    for sequence in range(SEQ_COUNT):
        # Load the numpy array you recorded
        res = np.load(os.path.join(DATA_PATH, action, f"{sequence}.npy"))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data (95% for training, 5% for a quick test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("🧠 Forging the Neural Network (This will take ~2 minutes)...")
model = build_lstm_model(len(ACTIONS))

# Train the model over 150 cycles
model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=1)

print("💾 Saving the Brain...")
model.save_weights('isl_weights.weights.h5')
print("✅ SUCCESS! 'isl_weights.h5' generated. You are ready to launch.")