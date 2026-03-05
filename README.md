# 🤟 SignSense AI - Real-Time Sign Language Deep Learning Engine

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-white?style=for-the-badge)

**SignSense AI** is a real-time, ultra-low latency Indian Sign Language (ISL) translation engine. Built with a microservice architecture, it captures spatial-temporal gestures via a browser-based frontend and streams them over WebSockets to a local deep learning backend for instant translation.

---

## ✨ Key Features
* **Zero-Cloud Latency:** Processes video and performs neural network inference entirely locally.
* **Temporal Memory:** Utilizes a strict 30-frame (1.0 second) rolling FIFO buffer to analyze the full trajectory of a sign, not just static poses.
* **Microservice Architecture:** Decouples heavy UI rendering from backend AI processing using bidirectional WebSockets.
* **Enterprise HUD Extension:** Includes a Manifest V3 Chrome Extension that injects live translated subtitles over video conferencing tools like Google Meet or Zoom.

## 🧠 Supported ISL Dictionary (Phase 1)
The current LSTM model has been trained on 30 sequences per word to recognize the following continuous conversational phrases:
| Greeting/Response | Polite Expressions |
| :--- | :--- |
| `Hello` | `Thank you` |
| `Good Morning` | `Sorry` |
| `How are you` | `I am fine` |

---

## 🏗️ System Architecture

1. **Vision Node (Frontend):** `index.html` or Chrome Extension utilizes Google's **MediaPipe** to extract 126 mathematical landmarks (x, y, z coordinates for both hands) from the webcam feed.
2. **The Bridge (Transport):** The coordinate JSON payload is streamed continuously via **WebSockets** to avoid HTTP overhead.
3. **The Engine (Backend):** A **FastAPI/Uvicorn** server catches the payload, manages the 30-frame memory buffer, and formats the tensor.
4. **The Brain (AI Inference):** A custom-trained **Keras LSTM** (Long Short-Term Memory) neural network analyzes the `(1, 30, 126)` tensor, predicts the phrase, and fires the text back to the UI.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
https://github.com/pradeep23g/SignLanguageInterpreter
cd SignLanguageInterpreter
```
### 2. Set Up the Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```
### 3. Install Dependencies
Ensure you are using Python 3.12+ and install the strict version requirements:
```bash
pip install -r requirements.txt
```
### 💻 Running the Application
Start the AI Backend Server
```Bash
python -m uvicorn main:app --reload
```
Wait for the Application startup complete message in your terminal.

### Launch the Frontend

Local Web App: Simply double-click index.html to open the experimental HUD in your browser.


