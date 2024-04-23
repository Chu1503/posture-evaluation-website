import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def pose_detector():
    camera = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Calculate angles
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Calculate accuracy 
                left_elbow_accuracy = 100 - abs(170 - left_elbow_angle)
                right_elbow_accuracy = 100 - abs(170 - right_elbow_angle)
                left_knee_accuracy = 100 - abs(170 - left_knee_angle)
                right_knee_accuracy = 100 - abs(130 - right_knee_angle)

                # Calculate overall accuracy
                overall_accuracy = (right_elbow_accuracy + left_elbow_accuracy + left_knee_accuracy + right_knee_accuracy) / 4

                # Visualize angle
                font_scale = 1.0
                frame_height, frame_width, _ = image.shape  # Get frame height and width
                cv2.rectangle(image, (0, 0), (300, 160), (0, 0, 0), -1)

                if 160 < left_elbow_angle < 180 and 160 < right_elbow_angle < 180 and 160 < left_knee_angle < 180 and 110 < right_knee_angle < 155:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                        )
                else:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                cv2.putText(image, f'{left_elbow_angle:.2f}', 
                            (int(left_elbow[0] * frame_width), int(left_elbow[1] * frame_height)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_elbow_angle:.2f}', 
                            (int(right_elbow[0] * frame_width), int(right_elbow[1] * frame_height)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{left_knee_angle:.2f}', 
                            (int(left_knee[0] * frame_width), int(left_knee[1] * frame_height)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_knee_angle:.2f}', 
                            (int(right_knee[0] * frame_width), int(right_knee[1] * frame_height)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 255, 255), 2, cv2.LINE_AA)
                
                # Display accuracies
                cv2.putText(image, f'Overall Accuracy: {overall_accuracy:.2f}%', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Elbow Accuracy: {left_elbow_accuracy:.2f}%', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Elbow Accuracy: {right_elbow_accuracy:.2f}%', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Knee Accuracy: {left_knee_accuracy:.2f}%', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Knee Accuracy: {right_knee_accuracy:.2f}%', (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
