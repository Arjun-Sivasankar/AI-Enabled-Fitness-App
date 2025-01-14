import cv2
import mediapipe as mp
import math

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    a, b, and c are tuples of (x, y, z) coordinates
    """
    a = (a.x, a.y)
    b = (b.x, b.y)
    c = (c.x, c.y)

    # Calculate vectors
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    # Calculate dot product and magnitude
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # Calculate angle
    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect poses
    results = pose.process(frame_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks for left arm
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Provide feedback
        if elbow_angle > 160:
            feedback = "Lower your arm for a full curl!"
        elif elbow_angle < 30:
            feedback = "Straighten your arm slowly!"
        else:
            feedback = "Good form!"

        # Display feedback on the frame
        cv2.putText(
            frame,
            feedback,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the angle for debugging purposes
        cv2.putText(
            frame,
            f"Elbow Angle: {int(elbow_angle)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Show the video frame
    cv2.imshow("Bicep Curl Feedback", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
