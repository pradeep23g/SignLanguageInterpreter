import cv2
import numpy as np
import os
import mediapipe as mp
import time

# 1. Hackathon 10 PM Evaluation Config
ACTIONS = ['Hello', 'How are you', 'I need help', 'Sorry', 'Thank you']
SEQ_COUNT = 30    # Number of videos per phrase
FRAME_COUNT = 20  # 0.8-second memory window to capture full phrases
DATA_PATH = "LSTM_Dataset"

# 2. Setup Folders
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# 3. Setup MediaPipe Engine
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def extract_keypoints(results):
    """Flattens 2 hands into a single 126-value array for the LSTM."""
    all_landmarks = np.zeros(126)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx < 2: # Limit to 2 hands max
                coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                # Hand 1 goes to indices 0-62, Hand 2 goes to 63-125
                all_landmarks[idx*63 : idx*63+63] = coords
    return all_landmarks

# 4. The Recording Loop
cap = cv2.VideoCapture(0)

print("🚀 INITIATING 45-FRAME DATA COLLECTION SPRINT...")
for action in ACTIONS:
    input(f"\n👉 Press ENTER when ready to record '{action}'...")
    time.sleep(2) # 2-second buffer to get your hands in position
    
    for sequence in range(SEQ_COUNT):
        sequence_data = []
        for frame_num in range(FRAME_COUNT):
            ret, frame = cap.read()
            # Convert color space for MediaPipe
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # UI Overlay
            cv2.putText(frame, f"Recording: {action} | Sequence: {sequence}/{SEQ_COUNT}", 
                        (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('SignSense - Phrase Collector', frame)
            
            # Extract math and append to current sequence
            sequence_data.append(extract_keypoints(results))
            cv2.waitKey(10) # 10ms delay captures natural motion speed
            
        # Save the 45-frame block
        npy_path = os.path.join(DATA_PATH, action, str(sequence))
        np.save(npy_path, np.array(sequence_data))
        
        print(f"✅ Saved sequence {sequence} for {action}")
        time.sleep(0.5) # Quick pause between recordings

cap.release()
cv2.destroyAllWindows()
print("\n🎉 DATA COLLECTION COMPLETE! Proceed to Training Phase.")