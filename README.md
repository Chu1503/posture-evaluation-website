# Dynamic_Posture_Evaluation

This repository contains datasets and scripts for the dynamic real-time automated evaluation and assessment of human body postural characteristics. The project focuses on comparing different pose estimation models to determine their accuracy, frame rates, and Euclidean distance measurements. The datasets included are used to analyze and evaluate the performance of YOLOv7 Pose and MediaPipe.

## Files Included

1. **Euclidean_Distance_Comparison.csv**
   - Contains data comparing the Euclidean distance between joint coordinates as detected by YOLOv7 Pose and MediaPipe. The dataset includes ground truth coordinates and the respective accuracy metrics for each joint.

2. **Frame_Rate_Comparison.csv**
   - Includes frame rate data comparing the performance of YOLOv7 Pose and MediaPipe. This file lists the frame rates for various sequences to assess the real-time capabilities of each model.

3. **Joint_Accuracy_Comparison.csv**
   - Provides accuracy comparison for specific joints detected in various images (e.g., Nose) by YOLOv7 Pose and MediaPipe. The dataset indicates the precision of each model in detecting joint positions.
