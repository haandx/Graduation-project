from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
app.static_folder = 'static'

# Initialize Mediapipe Pose module and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class Exercise:
    def __init__(self, pose_landmark1, pose_landmark2, pose_landmark3, correct_angle, incorrect_angle_threshold, feedback_text):
        self.pose_landmark1 = pose_landmark1
        self.pose_landmark2 = pose_landmark2
        self.pose_landmark3 = pose_landmark3
        self.correct_angle = correct_angle
        self.incorrect_angle_threshold = incorrect_angle_threshold
        self.feedback_text = feedback_text

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def process_frame(self, landmarks, image):
        angle = self.calculate_angle(
            [landmarks[self.pose_landmark1].x, landmarks[self.pose_landmark1].y],
            [landmarks[self.pose_landmark2].x, landmarks[self.pose_landmark2].y],
            [landmarks[self.pose_landmark3].x, landmarks[self.pose_landmark3].y]
        )

        if angle >= self.correct_angle - self.incorrect_angle_threshold and angle <= self.correct_angle + self.incorrect_angle_threshold:
            feedback = "Correct"
        else:
            feedback = "Incorrect"

        cv2.putText(image, f"Feedback: {feedback}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 130, 180), 2, cv2.LINE_AA)

        cv2.putText(image, f"Angle: {angle} degrees",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 130, 180), 2, cv2.LINE_AA)

        return feedback

# Define exercises with specific landmarks, angles, and feedback text
arm_raise = Exercise(
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    correct_angle=90.0,
    incorrect_angle_threshold=30.0,
    feedback_text="Arm Raise"
)

squat = Exercise(
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    correct_angle=120.0,
    incorrect_angle_threshold=20.0,
    feedback_text="Squat"
)
knee_flexion = Exercise(
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    correct_angle=90.0,
    incorrect_angle_threshold=20.0,
    feedback_text="Knee Flexion"
)

cap = cv2.VideoCapture(0)

def flip_image(frame):
    return cv2.flip(frame, 1)

def gen_frames(exercise):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = flip_image(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(60, 179, 113), thickness=4, circle_radius=4),
                mp_drawing.DrawingSpec(color=(60, 179, 113), thickness=4, circle_radius=4)
            )

            try:
                landmarks = results.pose_landmarks.landmark
                feedback = exercise.process_frame(landmarks, image)
            except:
                pass

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def exercise_feed(exercise):
    return Response(gen_frames(exercise), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('page2.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(arm_raise), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/arm_raise_feed')
def arm_raise_feed():
    return exercise_feed(arm_raise)

@app.route('/squat_feed')
def squat_feed():
    return exercise_feed(squat)
@app.route('/knee_flexion_feed')
def knee_flexion_feed():
    return exercise_feed(knee_flexion)

if __name__ == '__main__':
    app.run(debug=True)