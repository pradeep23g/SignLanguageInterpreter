from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from model_utils import build_lstm_model

app = FastAPI()

# --- SECURITY & CORS ---
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- AI CONFIGURATION ---
ACTIONS = np.array(['Hello', 'How are you', 'I need help', 'Sorry', 'Thank you']) 

# Build the architecture
model = build_lstm_model(len(ACTIONS))

# 🚨 THE BRAIN: Loaded with the strict Keras 3 filename
model.load_weights('isl_weights.weights.h5') 

# --- THE ENGINE WARM-UP (Fixes the 7-second delay) ---
print("🔥 Warming up the Deep Learning Engine (Compiling Graph)...")
dummy_data = np.zeros((1, 20, 126)) # Create a fake 45-frame sequence of zeros
model.predict(dummy_data, verbose=0) # Force TensorFlow to compile the graph now
print("✅ Engine Warmed Up! Ready for Real-Time Video.")
# ----------------------------------------------------

# --- WEBSOCKET ENGINE ---
sequence = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global sequence
    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data['landmarks']
            
            # --- THE LOGICAL GATE (Phantom Hand Fix) ---
            if not any(landmarks): 
                sequence = []      
                continue           
            # -------------------------------------------
            
            sequence.append(landmarks)
            sequence = sequence[-20:] # Maintain strict 45-frame rolling window
            
            # DEBUG: Watch the memory buffer fill up in your terminal
            print(f"👀 Tracking Hands... Buffer: {len(sequence)}/20", end="\r")
            
            # Only run the AI prediction when the memory buffer is full
            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                best_match_index = np.argmax(res)
                prediction = ACTIONS[best_match_index]
                confidence = float(res[best_match_index])
                
                # 🚨 THE THRESHOLD: Raised to 55% (0.55) to filter out background noise
                if confidence > 0.55: 
                    print(f"\n🧠 AI Guess: {prediction} | Confidence: {confidence*100:.1f}%")
                    
                    # Send the text to the frontend chat bubble
                    await websocket.send_json({
                        "prediction": prediction, 
                        "confidence": confidence
                    })
                    
                    # 🚨 THE SPAM FIX: Wipe the memory clean after a successful guess!
                    sequence = [] 
                    
    except Exception as e:
        print(f"\n⚠️ Connection closed or interrupted: {e}")