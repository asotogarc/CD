import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import math
from collections import deque
from flask import Flask, render_template, Response
import base64

app = Flask(__name__)

class EnhancedEmotionGazeTracker:
    def __init__(self):
        self.emotion_model = YOLO('best2.pt')
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.screen_rect = (50, 50, 590, 430)  # (x, y, w, h) of simulated screen
        self.heatmap = np.zeros((self.screen_rect[3] - self.screen_rect[1], 
                                 self.screen_rect[2] - self.screen_rect[0]), dtype=np.float32)
        self.gaze_history = deque(maxlen=30)  # Store last 30 gaze points
        self.emotion_history = deque(maxlen=10)  # Store last 10 emotions for stability
        self.emotion_threshold = 0.6  # Minimum confidence threshold for emotion detection
        self.zoom_factor = 1.0  # Initialize zoom factor
        self.optimal_face_size = 200  # Optimal face size in pixels

    def get_gaze_direction(self, face_landmarks, image):
        def get_eye_center(eye_landmarks):
            x_mean = sum([lm.x for lm in eye_landmarks]) / len(eye_landmarks)
            y_mean = sum([lm.y for lm in eye_landmarks]) / len(eye_landmarks)
            return (x_mean, y_mean)

        LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        left_eye = get_eye_center([face_landmarks.landmark[i] for i in LEFT_EYE])
        right_eye = get_eye_center([face_landmarks.landmark[i] for i in RIGHT_EYE])

        h, w, _ = image.shape
        left_eye = (int(left_eye[0] * w), int(left_eye[1] * h))
        right_eye = (int(right_eye[0] * w), int(right_eye[1] * h))

        gaze_direction_left = (left_eye[0] - w//2, left_eye[1] - h//2)
        gaze_direction_right = (right_eye[0] - w//2, right_eye[1] - h//2)

        return left_eye, right_eye, gaze_direction_left, gaze_direction_right

    def update_heatmap(self, gaze_point):
        x, y = gaze_point
        if (self.screen_rect[0] <= x < self.screen_rect[2] and 
            self.screen_rect[1] <= y < self.screen_rect[3]):
            self.heatmap[y - self.screen_rect[1], x - self.screen_rect[0]] += 1
        
        self.heatmap = cv2.GaussianBlur(self.heatmap, (15, 15), 0)

    def predict_gaze_zone(self):
        if len(self.gaze_history) < 5:
            return None
        
        recent_points = list(self.gaze_history)[-5:]
        avg_x = sum(p[0] for p in recent_points) / 5
        avg_y = sum(p[1] for p in recent_points) / 5

        screen_w = self.screen_rect[2] - self.screen_rect[0]
        screen_h = self.screen_rect[3] - self.screen_rect[1]

        if avg_x < self.screen_rect[0] + screen_w/3:
            h_zone = "izquierda"
        elif avg_x < self.screen_rect[0] + 2*screen_w/3:
            h_zone = "centro"
        else:
            h_zone = "derecha"

        if avg_y < self.screen_rect[1] + screen_h/3:
            v_zone = "top"
        elif avg_y < self.screen_rect[1] + 2*screen_h/3:
            v_zone = "medio"
        else:
            v_zone = "abajo"

        return f"{v_zone}-{h_zone}"

    def get_most_common_emotion(self):
        if not self.emotion_history:
            return "Unknown"

        emotion_counts = {}
        for emotion in self.emotion_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1

        return max(emotion_counts, key=emotion_counts.get)

    def adjust_zoom(self, face_landmarks, frame):
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        h, w, _ = frame.shape
        face_size = math.dist(
            (left_eye.x * w, left_eye.y * h),
            (right_eye.x * w, right_eye.y * h)
        )

        if face_size < self.optimal_face_size:
            self.zoom_factor = min(2.0, self.zoom_factor * 1.05)
        elif face_size > self.optimal_face_size * 1.2:
            self.zoom_factor = max(1.0, self.zoom_factor * 0.95)
        
        return self.zoom_factor

    def apply_zoom(self, frame, zoom_factor):
        h, w = frame.shape[:2]
        crop_size = int(min(h, w) / zoom_factor)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame):
        frame = self.apply_zoom(frame, self.zoom_factor)

        results = self.emotion_model(frame)
        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            detected_emotion = results[0].names[int(results[0].boxes[0].cls)]
            confidence = float(results[0].boxes[0].conf)
            if confidence >= self.emotion_threshold:
                self.emotion_history.append(detected_emotion)
        else:
            detected_emotion = "Unknown"

        stable_emotion = self.get_most_common_emotion()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            new_zoom_factor = self.adjust_zoom(face_landmarks, frame)
            if new_zoom_factor != self.zoom_factor:
                self.zoom_factor = new_zoom_factor
                return self.process_frame(frame)

            left_eye, right_eye, gaze_direction_left, gaze_direction_right = self.get_gaze_direction(face_landmarks, frame)
            
            cv2.arrowedLine(annotated_frame, left_eye, (left_eye[0] + gaze_direction_left[0] * 30, left_eye[1] + gaze_direction_left[1] * 30), (0, 255, 0), 2)
            cv2.arrowedLine(annotated_frame, right_eye, (right_eye[0] + gaze_direction_right[0] * 30, right_eye[1] + gaze_direction_right[1] * 30), (0, 255, 0), 2)
            
            gaze_point = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            self.update_heatmap(gaze_point)
            self.gaze_history.append(gaze_point)
            
            gaze_zone = self.predict_gaze_zone()
            
            cv2.putText(annotated_frame, f"Zoom: {self.zoom_factor:.2f}x", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Emocion estable: {stable_emotion}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if gaze_zone:
                cv2.putText(annotated_frame, f"Zona Pantalla: {gaze_zone}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(annotated_frame, (self.screen_rect[0], self.screen_rect[1]),
                      (self.screen_rect[2], self.screen_rect[3]), (255, 0, 0), 2)

        heatmap_colored = cv2.applyColorMap(cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(
            annotated_frame[self.screen_rect[1]:self.screen_rect[3], self.screen_rect[0]:self.screen_rect[2]],
            0.7,
            heatmap_colored,
            0.3,
            0
        )
        annotated_frame[self.screen_rect[1]:self.screen_rect[3], self.screen_rect[0]:self.screen_rect[2]] = heatmap_overlay

        return annotated_frame

tracker = EnhancedEmotionGazeTracker()
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = tracker.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)