from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import base64
import json
import asyncio
from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import tempfile
import os
from datetime import datetime
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Posture Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PostureAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculate angle between three points."""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def detect_squat_posture(self, landmarks) -> Dict:
        """Detect bad posture during squats, flag if knee goes beyond toe or back angle is not according to good sitting posture."""
        issues = []
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        # Check knee over toe
        if left_knee[0] > left_ankle[0]:
            issues.append("Knee going beyond toe")
        # Check back angle (shoulder-hip-knee)
        back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        # Good sitting posture: back angle 88-94
        if not (88 <= back_angle <= 94):
            issues.append(f"Back angle not in good sitting posture range: {back_angle:.1f}°")
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if knee_angle > 135:
            issues.append(f"Squat not deep enough: {knee_angle:.1f}°")
        good_posture = len(issues) == 0
        posture_quality = "Good Squat Posture" if good_posture else "Bad Squat Posture"
        return {
            "posture_type": "squat",
            "posture_quality": posture_quality,
            "issues": issues,
            "good_posture": good_posture,
            "back_angle": back_angle,
            "knee_angle": knee_angle
        }

    def detect_sitting_posture(self, landmarks) -> Dict:
        """Detect bad posture while sitting, including slouch detection via shoulder-hip verticality and symmetry."""
        issues = []
        nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y])
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        # Neck angle (forward head posture)
        neck_vector = nose - ((left_shoulder + right_shoulder) / 2)
        neck_angle = np.degrees(np.arctan2(neck_vector[1], neck_vector[0]))
        # Back straightness (shoulder-hip alignment, both sides)
        back_vector_left = left_shoulder - left_hip
        back_vector_right = right_shoulder - right_hip
        back_angle_left = np.degrees(np.arctan2(back_vector_left[1], back_vector_left[0]))
        back_angle_right = np.degrees(np.arctan2(back_vector_right[1], back_vector_right[0]))
        avg_back_angle = (abs(back_angle_left) + abs(back_angle_right)) / 2
        # Shoulder angle (angle between left-right shoulder and horizontal axis)
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        shoulder_angle_abs = abs(shoulder_angle)
        good_posture = True
        # Good sitting posture: back angle 88-94, neck angle 68-100, shoulders level (angle ~0)
        if not (88 <= avg_back_angle <= 94):
            issues.append(f"Back angle out of range: {avg_back_angle:.1f}°")
            good_posture = False
        if not (68 <= abs(neck_angle) <= 100):
            issues.append(f"Neck angle out of range: {abs(neck_angle):.1f}°")
            good_posture = False
        if not (175<=shoulder_angle_abs <= 185):
            issues.append(f"Shoulders not level: {shoulder_angle_abs:.1f}° from horizontal")
            good_posture = False
        # Slouch detection: large difference between left/right back angles or both out of range in same direction
        back_angle_diff = abs(abs(back_angle_left) - abs(back_angle_right))
        if back_angle_diff > 9:
            issues.append(f"Slouch detected: left/right back angle difference {back_angle_diff:.1f}°")
            good_posture = False
        
        posture_quality = "Good Sitting Posture" if good_posture else "Bad Sitting Posture"
        return {
            "posture_type": "sitting",
            "posture_quality": posture_quality,
            "issues": issues,
            "good_posture": good_posture,
            "neck_angle": abs(neck_angle),
            "back_angle": avg_back_angle,
            "shoulder_angle": shoulder_angle_abs,
            "back_angle_left": back_angle_left,
            "back_angle_right": back_angle_right
        }

    def detect_standing_posture(self, landmarks) -> Dict:
        """Enhanced: Detect bad posture while standing using all body points and robust verticality check."""
        issues = []
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y])
        # Symmetry
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_diff = abs(left_hip[1] - right_hip[1])
        # Back angle: angle between hip-shoulder vector and vertical axis (y negative direction)
        vertical = np.array([0, -1])
        def angle_with_vertical(v):
            v_norm = v / (np.linalg.norm(v) + 1e-8)
            dot = np.dot(v_norm, vertical)
            dot = np.clip(dot, -1.0, 1.0)
            return np.degrees(np.arccos(dot))
        back_angle_left = angle_with_vertical(left_shoulder - left_hip)
        back_angle_right = angle_with_vertical(right_shoulder - right_hip)
        # Leg straightness (hip-knee-ankle angles)
        left_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        # Head/neck alignment
        neck_vector = nose - ((left_shoulder + right_shoulder) / 2)
        neck_angle = np.degrees(np.arctan2(neck_vector[1], neck_vector[0]))
        # Good posture: back angles <10°, neck angle 80-100, leg angles > 165, shoulders/hips level
        good_posture = True
        if not (0 <= back_angle_left <= 10):
            issues.append(f"Left back angle out of range (not upright): {back_angle_left:.1f}°")
            good_posture = False
        if not (0 <= back_angle_right <= 10):
            issues.append(f"Right back angle out of range (not upright): {back_angle_right:.1f}°")
            good_posture = False
        if not (80 <= abs(neck_angle) <= 100):
            issues.append(f"Neck angle out of range: {abs(neck_angle):.1f}°")
            good_posture = False
        if left_leg_angle < 165:
            issues.append(f"Left leg not straight: {left_leg_angle:.1f}°")
            good_posture = False
        if right_leg_angle < 165:
            issues.append(f"Right leg not straight: {right_leg_angle:.1f}°")
            good_posture = False
        if shoulder_diff > 0.05:
            issues.append(f"Shoulders not level: {shoulder_diff:.2f}")
            good_posture = False
        if hip_diff > 0.05:
            issues.append(f"Hips not level: {hip_diff:.2f}")
            good_posture = False
        posture_quality = "Good Standing Posture" if good_posture else "Bad Standing Posture"
        return {
            "posture_type": "standing",
            "posture_quality": posture_quality,
            "issues": issues,
            "good_posture": good_posture,
            "neck_angle": abs(neck_angle),
            "back_angle_left": back_angle_left,
            "back_angle_right": back_angle_right,
            "left_leg_angle": left_leg_angle,
            "right_leg_angle": right_leg_angle,
            "shoulder_diff": shoulder_diff,
            "hip_diff": hip_diff
        }

    def analyze_frame(self, frame: np.ndarray, posture_type: str = "auto") -> Dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return {
                "detected": False,
                "message": "No pose detected"
            }
        landmarks = results.pose_landmarks.landmark
        # Use more body points for robust detection
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        # Calculate vertical and horizontal distances
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        avg_knee_y = (left_knee.y + right_knee.y) / 2
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_x = (left_hip.x + right_hip.x) / 2
        avg_knee_x = (left_knee.x + right_knee.x) / 2
        avg_ankle_x = (left_ankle.x + right_ankle.x) / 2
        avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        # Calculate body height (shoulder to ankle)
        body_height = abs(avg_shoulder_y - avg_ankle_y)
        # Calculate hip-to-ankle and knee-to-ankle ratios
        hip_ankle_dist = abs(avg_hip_y - avg_ankle_y)
        knee_ankle_dist = abs(avg_knee_y - avg_ankle_y)
        # Calculate knee flexion (bent or straight)
        knee_flexion = max(abs(left_knee.y - left_ankle.y), abs(right_knee.y - right_ankle.y))
        # Calculate torso angle (shoulder-hip-ankle)
        torso_angle = np.degrees(np.arctan2(avg_shoulder_y - avg_hip_y, avg_shoulder_x - avg_hip_x))
        # Calculate head/neck position
        neck_forward = nose.x - avg_shoulder_x
        # Heuristic: use all these for robust detection
        if posture_type == "auto":
            # Standing: hips and ankles nearly aligned vertically, knees close to hips, body height is large, knees mostly straight
            if (
                hip_ankle_dist / body_height < 0.18 and
                knee_ankle_dist / body_height < 0.25 and
                knee_flexion < 0.12 and
                abs(torso_angle) < 25 and
                abs(neck_forward) < 0.10
            ):
                posture_type = "standing"
            # Squat: hips much lower than knees, knees bent, body height reduced
            elif (
                avg_hip_y > avg_knee_y + 0.10 and
                knee_flexion > 0.18 and
                hip_ankle_dist / body_height > 0.22
            ):
                posture_type = "squat"
            # Sitting: hips above knees, knees bent, body height reduced, but not as much as squat
            else:
                posture_type = "sitting"
        if posture_type == "squat":
            analysis = self.detect_squat_posture(landmarks)
        elif posture_type == "standing":
            analysis = self.detect_standing_posture(landmarks)
        else:
            analysis = self.detect_sitting_posture(landmarks)
        analysis["landmarks"] = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in landmarks
        ]
        analysis["detected"] = True
        return analysis

# Global analyzer instance
analyzer = PostureAnalyzer()

# Pydantic models
class PostureAnalysisResponse(BaseModel):
    detected: bool
    posture_type: Optional[str] = None
    issues: List[str] = []
    good_posture: bool = False
    landmarks: Optional[List[Dict]] = None
    message: Optional[str] = None

class VideoAnalysisResponse(BaseModel):
    total_frames: int
    analyzed_frames: int
    bad_posture_frames: int
    good_posture_percentage: float
    frame_analyses: List[PostureAnalysisResponse]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Posture Detection API", "version": "1.0.0"}

@app.post("/analyze/image", response_model=PostureAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    posture_type: str = "auto"
):
    """Analyze posture from uploaded image."""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze posture
        result = analyzer.analyze_frame(frame, posture_type)
        
        return PostureAnalysisResponse(**result)
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    posture_type: str = "auto",
    sample_rate: int = 5  # Analyze every 5th frame
):
    """Analyze posture from uploaded video."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # Process video
        cap = cv2.VideoCapture(tmp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_analyses = []
        analyzed_frames = 0
        bad_posture_frames = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to avoid processing every frame
            if frame_count % sample_rate == 0:
                result = analyzer.analyze_frame(frame, posture_type)
                frame_analyses.append(PostureAnalysisResponse(**result))
                analyzed_frames += 1
                
                if result.get("detected", False) and not result.get("good_posture", False):
                    bad_posture_frames += 1
            
            frame_count += 1
        
        cap.release()
        os.unlink(tmp_file_path)  # Clean up temp file
        
        good_posture_percentage = ((analyzed_frames - bad_posture_frames) / analyzed_frames * 100) if analyzed_frames > 0 else 0
        
        return VideoAnalysisResponse(
            total_frames=total_frames,
            analyzed_frames=analyzed_frames,
            bad_posture_frames=bad_posture_frames,
            good_posture_percentage=good_posture_percentage,
            frame_analyses=frame_analyses
        )
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time posture analysis with keepalive and robust error handling."""
    await manager.connect(websocket)
    try:
        while True:
            try:
                # Wait for frame data from client with timeout (e.g., 15 seconds)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15)
                frame_data = json.loads(data)
                # Decode base64 image
                image_data = base64.b64decode(frame_data["image"].split(",")[1])
                image = Image.open(io.BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # Analyze posture
                result = analyzer.analyze_frame(frame, frame_data.get("posture_type", "auto"))
                # Send result back
                await manager.send_personal_message(json.dumps(result), websocket)
            except asyncio.TimeoutError:
                # If no frame received in time, send keepalive or warning
                await manager.send_personal_message(json.dumps({"warning": "No frame received in 15 seconds. Send frames to keep connection alive."}), websocket)
                continue
            except Exception as e:
                logger.error(f"WebSocket frame error: {str(e)}")
                await manager.send_personal_message(json.dumps({"error": str(e)}), websocket)
                continue
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.send_personal_message(json.dumps({"error": str(e)}), websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
