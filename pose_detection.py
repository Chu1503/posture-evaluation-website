import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def detect_pose(selected_pose="Warrior Pose"):
    # Initialize variables
    prev_time = 0
    fps = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
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

                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Calculate accuracy for left elbow angle
                left_elbow_accuracy = 100 - abs(170 - left_elbow_angle)

                # Calculate accuracy for right elbow angle
                right_elbow_accuracy = 100 - abs(170 - right_elbow_angle)

                # Calculate accuracy for left knee angle
                left_knee_accuracy = 100 - abs(170 - left_knee_angle)

                # Calculate accuracy for right knee angle
                right_knee_accuracy = 100 - abs(120 - right_knee_angle)

                # Calculate overall accuracy
                overall_accuracy = (right_elbow_accuracy + left_elbow_accuracy + left_knee_accuracy + right_knee_accuracy) / 4

                # Visualize angle
                font_scale = 1.0
                
                cv2.rectangle(image, (0, 0), (300, 200), (0, 0, 0), -1)


                cv2.putText(image, f'{left_elbow_angle:.2f}', 
                            tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_elbow_angle:.2f}', 
                            tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{left_knee_angle:.2f}', 
                            tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_knee_angle:.2f}', 
                            tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)

                # Display accuracies
                cv2.putText(image, f'Overall Accuracy: {overall_accuracy:.2f}%', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Elbow Accuracy: {left_elbow_accuracy:.2f}%', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Elbow Accuracy: {right_elbow_accuracy:.2f}%', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Knee Accuracy: {left_knee_accuracy:.2f}%', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Knee Accuracy: {right_knee_accuracy:.2f}%', (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections
            if (160 < left_elbow_angle < 180):
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                    )
            
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                    )

            # Calculate and display FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            fps = 1 / elapsed_time
            prev_time = current_time

            cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Yoga Pose Estimation - Virabhadrasana', image)

            key = cv2.waitKey(10)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()