from flask import Flask, render_template, request
from pose_detection import detect_pose

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_pose_detection', methods=['POST'])
def run_pose_detection():
    selected_pose = request.form['selected_pose']  # Get the selected pose from the form data
    detect_pose(selected_pose)  # Pass the selected pose to detect_pose function
    return 'Pose detection started.'

if __name__ == '__main__':
    app.run(debug=True)
