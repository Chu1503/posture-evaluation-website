from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np

def index(request):
    return render(request, 'yoga_pose_app/index.html')

class VideoStream:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __del__(self):
        self.video.release()

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get image dimensions
        height, width, _ = image.shape

        # Draw landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            def get_coords(landmark):
                return int(landmark.x * width), int(landmark.y * height)

            # Coordinates for landmarks
            left_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            left_elbow = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value])
            left_wrist = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])

            right_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            right_elbow = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])
            right_wrist = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])

            left_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])
            left_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])
            left_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

            right_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value])
            right_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])
            right_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])

            # Calculate angles
            left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            # Draw landmarks and angles on image
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Add text to image
            def add_text(coords, angle):
                cv2.putText(image, f'{angle:.2f}', coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            add_text(left_elbow, left_elbow_angle)
            add_text(right_elbow, right_elbow_angle)
            add_text(left_knee, left_knee_angle)
            add_text(right_knee, right_knee_angle)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def video_feed(request):
    if 'close' in request.GET:
        return StreamingHttpResponse(None, content_type='multipart/x-mixed-replace; boundary=frame')

    return StreamingHttpResponse(gen(VideoStream()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')