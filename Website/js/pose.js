const videoElement = document.createElement('video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let camera = null;
let pose = null;

// Function to calculate the angle between three points
function calculateAngle(a, b, c) {
  const AB = { x: b.x - a.x, y: b.y - a.y };
  const BC = { x: c.x - b.x, y: c.y - b.y };

  const dotProduct = AB.x * BC.x + AB.y * BC.y;
  const magnitudeAB = Math.sqrt(AB.x * AB.x + AB.y * AB.y);
  const magnitudeBC = Math.sqrt(BC.x * BC.x + BC.y * BC.y);

  const angle = Math.acos(dotProduct / (magnitudeAB * magnitudeBC));
  return angle * (180.0 / Math.PI);  // Convert radians to degrees
}

// Function to start pose detection
function startPoseDetection() {
  document.body.classList.add('loaded');

  // Ensure that the video element has a stream source
  navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    videoElement.srcObject = stream;
    videoElement.play();
  }).catch(error => {
    console.error("Error accessing webcam: ", error);
    alert("Could not access webcam. Please check permissions.");
  });

  if (!pose) {
    pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.2/${file}`,
    });
    pose.onResults(onResultsPose);
  }

  if (!camera) {
    camera = new Camera(videoElement, {
      onFrame: async () => {
        await pose.send({ image: videoElement });
      },
      width: 1080,
      height: 720
    });
    camera.start();
  }

  document.getElementById('startButton').disabled = true;
  document.getElementById('stopButton').disabled = false;
}

// Function to stop pose detection
function stopPoseDetection() {
  if (camera) {
    // Manually stop the video stream if the Camera instance does not have a stop method
    const stream = videoElement.srcObject;
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());  // Stop each track in the stream
    }
    videoElement.srcObject = null;  // Remove the video stream
    camera = null;
  }

  if (pose) {
    pose.close();  // Close the Pose instance if necessary
  }

  document.getElementById('startButton').disabled = false;
  document.getElementById('stopButton').disabled = true;

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);  // Clear the canvas
}


// Function to draw connectors with conditional color
function drawConnectorsConditional(ctx, landmarks, connections) {
  connections.forEach(([fromIndex, toIndex]) => {
    const from = landmarks[fromIndex];
    const to = landmarks[toIndex];

    // Default color is white
    let color = 'rgba(255,255,255,0.5)';

    // Check angles and update color accordingly
    let allConditionsMet = true;

    // Left Elbow Angle (0 to 30 degrees)
    if (landmarks[POSE_LANDMARKS.LEFT_ELBOW] && landmarks[POSE_LANDMARKS.LEFT_WRIST] && landmarks[POSE_LANDMARKS.LEFT_SHOULDER]) {
      const leftElbowAngle = calculateAngle(
        landmarks[POSE_LANDMARKS.LEFT_WRIST],
        landmarks[POSE_LANDMARKS.LEFT_ELBOW],
        landmarks[POSE_LANDMARKS.LEFT_SHOULDER]
      );
      if (leftElbowAngle < 0 || leftElbowAngle > 30) {
        allConditionsMet = false;
      }
    } else {
      allConditionsMet = false;
    }

    // Right Elbow Angle (40 to 50 degrees)
    if (landmarks[POSE_LANDMARKS.RIGHT_ELBOW] && landmarks[POSE_LANDMARKS.RIGHT_WRIST] && landmarks[POSE_LANDMARKS.RIGHT_SHOULDER]) {
      const rightElbowAngle = calculateAngle(
        landmarks[POSE_LANDMARKS.RIGHT_WRIST],
        landmarks[POSE_LANDMARKS.RIGHT_ELBOW],
        landmarks[POSE_LANDMARKS.RIGHT_SHOULDER]
      );
      if (rightElbowAngle < 40 || rightElbowAngle > 50) {
        allConditionsMet = false;
      }
    } else {
      allConditionsMet = false;
    }

    // Left Knee Angle (90 to 120 degrees)
    if (landmarks[POSE_LANDMARKS.LEFT_HIP] && landmarks[POSE_LANDMARKS.LEFT_KNEE] && landmarks[POSE_LANDMARKS.LEFT_ANKLE]) {
      const leftKneeAngle = calculateAngle(
        landmarks[POSE_LANDMARKS.LEFT_HIP],
        landmarks[POSE_LANDMARKS.LEFT_KNEE],
        landmarks[POSE_LANDMARKS.LEFT_ANKLE]
      );
      if (leftKneeAngle < 90 || leftKneeAngle > 120) {
        allConditionsMet = false;
      }
    } else {
      allConditionsMet = false;
    }

    // Right Knee Angle (130 to 160 degrees)
    if (landmarks[POSE_LANDMARKS.RIGHT_HIP] && landmarks[POSE_LANDMARKS.RIGHT_KNEE] && landmarks[POSE_LANDMARKS.RIGHT_ANKLE]) {
      const rightKneeAngle = calculateAngle(
        landmarks[POSE_LANDMARKS.RIGHT_HIP],
        landmarks[POSE_LANDMARKS.RIGHT_KNEE],
        landmarks[POSE_LANDMARKS.RIGHT_ANKLE]
      );
      if (rightKneeAngle < 130 || rightKneeAngle > 160) {
        allConditionsMet = false;
      }
    } else {
      allConditionsMet = false;
    }

    if (allConditionsMet) {
      color = 'green';
    }

    ctx.beginPath();
    ctx.moveTo(from.x * canvasElement.width, from.y * canvasElement.height);
    ctx.lineTo(to.x * canvasElement.width, to.y * canvasElement.height);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function onResultsPose(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
  // Draw connectors with conditional color
  drawConnectorsConditional(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS);
  
  drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', fillColor: '#FF0000' });

  // Calculate angles
  const landmarks = results.poseLandmarks;
  const leftElbowAngle = calculateAngle(landmarks[POSE_LANDMARKS.LEFT_WRIST], landmarks[POSE_LANDMARKS.LEFT_ELBOW], landmarks[POSE_LANDMARKS.LEFT_SHOULDER]);
  const rightElbowAngle = calculateAngle(landmarks[POSE_LANDMARKS.RIGHT_WRIST], landmarks[POSE_LANDMARKS.RIGHT_ELBOW], landmarks[POSE_LANDMARKS.RIGHT_SHOULDER]);
  const leftKneeAngle = calculateAngle(landmarks[POSE_LANDMARKS.LEFT_HIP], landmarks[POSE_LANDMARKS.LEFT_KNEE], landmarks[POSE_LANDMARKS.LEFT_ANKLE]);
  const rightKneeAngle = calculateAngle(landmarks[POSE_LANDMARKS.RIGHT_HIP], landmarks[POSE_LANDMARKS.RIGHT_KNEE], landmarks[POSE_LANDMARKS.RIGHT_ANKLE]);

  // Display angles near the corresponding joints
  canvasCtx.font = "18px Arial";
  canvasCtx.fillStyle = "white";

  // Left Elbow
  if (landmarks[POSE_LANDMARKS.LEFT_ELBOW]) {
    const leftElbow = landmarks[POSE_LANDMARKS.LEFT_ELBOW];
    canvasCtx.fillText(`${leftElbowAngle.toFixed(2)}°`, leftElbow.x * canvasElement.width + 10, leftElbow.y * canvasElement.height);
  }

  // Right Elbow
  if (landmarks[POSE_LANDMARKS.RIGHT_ELBOW]) {
    const rightElbow = landmarks[POSE_LANDMARKS.RIGHT_ELBOW];
    canvasCtx.fillText(`${rightElbowAngle.toFixed(2)}°`, rightElbow.x * canvasElement.width + 10, rightElbow.y * canvasElement.height);
  }

  // Left Knee
  if (landmarks[POSE_LANDMARKS.LEFT_KNEE]) {
    const leftKnee = landmarks[POSE_LANDMARKS.LEFT_KNEE];
    canvasCtx.fillText(`${leftKneeAngle.toFixed(2)}°`, leftKnee.x * canvasElement.width + 10, leftKnee.y * canvasElement.height);
  }

  // Right Knee
  if (landmarks[POSE_LANDMARKS.RIGHT_KNEE]) {
    const rightKnee = landmarks[POSE_LANDMARKS.RIGHT_KNEE];
    canvasCtx.fillText(`${rightKneeAngle.toFixed(2)}°`, rightKnee.x * canvasElement.width + 10, rightKnee.y * canvasElement.height);
  }

  canvasCtx.restore();
}

// Attach event listeners to start/stop buttons
document.getElementById('startButton').addEventListener('click', startPoseDetection);
document.getElementById('stopButton').addEventListener('click', stopPoseDetection);