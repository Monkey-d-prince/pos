# Posture Detection App

A full-stack application for real-time, image, and video-based human posture detection using MediaPipe, OpenCV, FastAPI (backend), and React (frontend).

## Features

- **Real-Time Posture Detection:** Uses webcam and WebSocket for instant feedback.
- **Image & Video Upload (Coming Soon):** Analyze posture from uploaded images and videos.
- **Posture Types:** Automatically detects and analyzes sitting, standing, and squat postures.
- **Detailed Feedback:** Provides posture quality, detected issues, and visual overlays.

---

## Backend

- **Framework:** FastAPI
- **Pose Estimation:** MediaPipe
- **Image Processing:** OpenCV, Pillow
- **API Endpoints:**
  - `POST /analyze/image` - Analyze posture from an uploaded image.
  - `POST /analyze/video` - Analyze posture from an uploaded video.
  - `GET /health` - Health check.
  - `GET /` - API info.
  - `WS /ws/realtime` - WebSocket for real-time webcam posture analysis.

### Setup & Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend

- **Framework:** React
- **Routing:** React Router
- **UI:** Custom CSS, Material UI (optional)
- **Pages:**
  - Real-Time Detection (Webcam)
  - Image Upload (Coming Soon)
  - Video Upload (Coming Soon)

### Setup & Run

```bash
cd frontend
npm install
npm start
```

---

## Folder Structure

```
backend/
  main.py
  requirements.txt
frontend/
  src/
    App.js
    components/
      Header.js
      RealTime.js
  public/
    index.html
  package.json
```

---

## How It Works

- **Real-Time:** The frontend captures webcam frames and sends them via WebSocket to the backend. The backend analyzes the frame and returns posture feedback and landmarks.
- **Image/Video:** (Coming soon) Upload an image or video for batch analysis.

---

## License

MIT License

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
- [React](https://react.dev/)
