import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import time
import os
from collections import Counter # To count feedback occurrences

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Drawing Specs ---
# (Keep Drawing Specs as before)
left_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3)
left_connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
right_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)
right_connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
other_connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

 # Angle thresholds
thresholds = {
        "pushup": {"up": 150, "down": 90},
        "squat": {"up": 165, "down": 90},
        "plank": {"straight": 160, "tolerance": 15}
    }

# --- Helper Functions ---
# (Keep calculate_angle, draw_landmarks_custom, display_status_box, draw_progress_bar as before)
def calculate_angle(a, b, c):
    """Calculates angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    try:
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        if angle > 180.0:
            angle = 360 - angle
    except Exception as e:
        angle = 0
    return angle

def draw_landmarks_custom(image, landmarks, connections, left_spec, right_spec, other_spec):
    """Draws landmarks with different colors for left, right, and torso."""
    left_connections_indices = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER)]
    right_connections_indices = [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                                 (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                                 (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                                 (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                                 (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER)]
    torso_connections_indices = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                 (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                                 (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                                 (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)]

    mp_drawing.draw_landmarks(image, landmarks, left_connections_indices,
                              landmark_drawing_spec=left_spec,
                              connection_drawing_spec=left_connection_drawing_spec)
    mp_drawing.draw_landmarks(image, landmarks, right_connections_indices,
                              landmark_drawing_spec=right_spec,
                              connection_drawing_spec=right_connection_drawing_spec)
    mp_drawing.draw_landmarks(image, landmarks, torso_connections_indices,
                              connection_drawing_spec=other_spec)

def display_status_box(image, reps, feedback, timer=None):
    """Displays reps, feedback, and optional timer."""
    box_x, box_y, box_w, box_h = 10, 10, 250, 100 if timer is None else 130
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (245, 117, 16), -1)
    cv2.putText(image, 'REPS', (box_x + 15, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(reps), (box_x + 20, box_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'FEEDBACK', (box_x + 100, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, feedback, (box_x + 100, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if timer is not None:
        cv2.putText(image, 'TIMER', (box_x + 15, box_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{int(timer)}s", (box_x + 100, box_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def draw_progress_bar(image, angle, angle_down, angle_up):
    """Draws a progress bar based on the angle."""
    image_height, image_width, _ = image.shape
    progress = np.interp(angle, [angle_down, angle_up], [100, 0])

    # bar_x, bar_y, bar_w, bar_h = image_width - 60, image_height - 250, 40, 200
    # cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), 2)
    # fill_height = int((progress / 100) * bar_h)
    # cv2.rectangle(image, (bar_x, bar_y + bar_h - fill_height), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
    # y_up_threshold = bar_y
    # y_down_threshold = bar_y + bar_h
    # cv2.line(image, (bar_x - 5, y_up_threshold), (bar_x + bar_w + 5, y_up_threshold), (0, 255, 255), 2)
    # cv2.line(image, (bar_x - 5, y_down_threshold), (bar_x + bar_w + 5, y_down_threshold), (0, 255, 255), 2)
    # cv2.putText(image, f'{int(progress)}%', (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(image, (50, 200), (100, 400), (255, 0, 0), 2)
    cv2.rectangle(image, (50, int(400 - (progress * 2))), (100, 400), (255, 0, 0), -1)
    cv2.putText(image, f'{int(progress)}%', (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# --- Exercise Processing Logic ---
def process_video(video_path, exercise_type):
    """Processes video, yields processed frame and metrics dict."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        yield None, None # Yield None for frame and metrics
        return

    # Exercise specific variables
    rep_count = 0
    exercise_state = None
    feedback = f"Start {exercise_type.capitalize()}"
    start_time = None
    hold_duration = 0
    frame_metrics = {} # Initialize metrics dict for each frame


    angle_up = thresholds.get(exercise_type, {}).get("up", 180)
    angle_down = thresholds.get(exercise_type, {}).get("down", 90)
    plank_straight = thresholds.get("plank", {}).get("straight", 160)
    plank_tolerance = thresholds.get("plank", {}).get("tolerance", 15)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current time in video (seconds)
            image_height, image_width, _ = frame.shape
            frame_metrics = { # Reset metrics for current frame
                 "timestamp": timestamp,
                 "avg_angle": None, "left_angle": None, "right_angle": None,
                 "current_feedback": feedback, "rep_count": rep_count,
                 "state": exercise_state, "hold_duration": hold_duration
            }

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            frame.flags.writeable = True

            avg_angle = 0 # Reset avg_angle for the frame

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                draw_landmarks_custom(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      left_landmark_drawing_spec, right_landmark_drawing_spec, other_connection_drawing_spec)

                try:
                    # --- Calculations based on Exercise ---
                    if exercise_type == "pushup":
                        # (Get coordinates as before)
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height]

                        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        avg_angle = (left_angle + right_angle) / 2
                        frame_metrics.update({"avg_angle": avg_angle, "left_angle": left_angle, "right_angle": right_angle})

                        # (Display angles as before)
                        cv2.putText(frame, f"L: {int(left_angle)}", (int(left_elbow[0]) - 60, int(left_elbow[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"R: {int(right_angle)}", (int(right_elbow[0]) + 10, int(right_elbow[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                        # (Rep Counting & Feedback logic as before)
                        if avg_angle > angle_up:
                            if exercise_state == 'down': feedback = "Up"
                            exercise_state = 'up'
                        elif avg_angle < angle_down:
                            if exercise_state == 'up':
                                rep_count += 1
                                feedback = "Down"
                            exercise_state = 'down'

                        if exercise_state == 'up' and avg_angle < angle_up - 10: feedback = "Extend Arms Fully"
                        elif exercise_state == 'down' and avg_angle > angle_down + 10: feedback = "Go Lower"

                    elif exercise_type == "squat":
                        # (Get coordinates as before)
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]
                        # Add right side calculation for potential asymmetry check later
                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
                        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height]
                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height]

                        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                        avg_angle = (left_angle + right_angle) / 2 # Use average for main logic
                        frame_metrics.update({"avg_angle": avg_angle, "left_angle": left_angle, "right_angle": right_angle})

                        # (Display angle as before - maybe show avg or both L/R)
                        cv2.putText(frame, f"L:{int(left_angle)} R:{int(right_angle)}", (int(left_knee[0]) + 10, int(left_knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                        # (Rep Counting & Feedback logic as before)
                        if avg_angle > angle_up:
                            if exercise_state == 'down': feedback = "Stand Up"
                            exercise_state = 'up'
                        elif avg_angle < angle_down:
                            if exercise_state == 'up':
                                rep_count += 1
                                feedback = "Squat Down"
                            exercise_state = 'down'

                        if exercise_state == 'up' and avg_angle < angle_up - 10: feedback = "Stand Straighter"
                        elif exercise_state == 'down' and avg_angle > angle_down + 15: feedback = "Go Deeper"


                    elif exercise_type == "plank":
                        # (Get coordinates as before)
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]
                        avg_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                        frame_metrics.update({"avg_angle": avg_angle})

                        # (Display angle as before)
                        cv2.putText(frame, f"{int(avg_angle)}", (int(left_hip[0]) + 10, int(left_hip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        # (Hold Timer & Feedback logic as before)
                        if avg_angle > plank_straight - plank_tolerance and avg_angle < plank_straight + plank_tolerance:
                            if exercise_state != 'hold':
                                start_time = time.time()
                                feedback = "Hold Straight"
                            exercise_state = 'hold'
                            # Calculate current hold duration for this frame
                            current_hold_time = time.time() - start_time if start_time else 0
                            hold_duration = current_hold_time # Update total hold duration continuously for display
                        else:
                            if exercise_state == 'hold':
                                start_time = None # Reset timer start if form breaks
                            exercise_state = 'adjust'
                            if avg_angle < plank_straight - plank_tolerance:
                                feedback = "Lift Hips"
                            else:
                                feedback = "Lower Hips"
                            hold_duration = 0 # Reset display duration if form is wrong

                        frame_metrics.update({"hold_duration": hold_duration}) # Update metric

                except (IndexError, KeyError, AttributeError) as e:
                    feedback = "Landmarks missing"
                except Exception as e:
                    feedback = "Processing Error"

                # Update metrics dict with latest values for this frame
                frame_metrics.update({
                    "current_feedback": feedback,
                    "rep_count": rep_count,
                    "state": exercise_state
                })

                # Display status box
                timer_display = hold_duration if exercise_type == "plank" else None
                display_status_box(frame, rep_count, feedback, timer=timer_display)

                # Draw progress bar
                if exercise_type in ["pushup", "squat"]:
                    draw_progress_bar(frame, avg_angle, angle_down, angle_up)

            else: # No landmarks detected
                feedback = "No person detected"
                display_status_box(frame, rep_count, feedback)
                frame_metrics.update({"current_feedback": feedback, "state": "no_person"})


            # Yield the processed frame and the collected metrics for this frame
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_metrics

    cap.release()

# --- Report Generation Function --- (JUST A DUMMY VERSION NOW)
def generate_workout_report(collected_data, exercise_type):
    report = f"## Workout Summary: {exercise_type.capitalize()}\n\n"
    if not collected_data:
        return report + "No data collected to generate a report."

    # --- General Stats ---
    total_duration = collected_data[-1]['timestamp'] - collected_data[0]['timestamp'] if len(collected_data) > 1 else 0
    report += f"*   **Total Video Duration Analyzed:** {total_duration:.1f} seconds\n"

    # --- Exercise Specific Stats & Feedback ---
    feedback_counter = Counter(d['current_feedback'] for d in collected_data)
    common_corrections = {fb: count for fb, count in feedback_counter.items() if fb not in [
        "Up", "Down", "Good - Up", "Good - Down", "Hold Straight", f"Start {exercise_type.capitalize()}",
        "No person detected", "Landmarks missing", "Processing Error", "Stand Up", "Squat Down"
    ]}

    if exercise_type in ["pushup", "squat"]:
        total_reps = max(d['rep_count'] for d in collected_data)
        report += f"*   **Total Reps:** {total_reps}\n"
        if total_duration > 0 and total_reps > 0:
             report += f"*   **Average Pace:** {total_reps / total_duration * 60:.1f} reps/minute\n"

        # Analyze angles during reps
        down_angles = [d['avg_angle'] for d in collected_data if d['state'] == 'down' and d['avg_angle'] is not None]
        up_angles = [d['avg_angle'] for d in collected_data if d['state'] == 'up' and d['avg_angle'] is not None]
        avg_depth = np.mean(down_angles) if down_angles else None
        avg_height = np.mean(up_angles) if up_angles else None

        if avg_depth is not None:
            report += f"*   **Average Depth Angle:** {avg_depth:.1f}°\n"
        if avg_height is not None:
            report += f"*   **Average Peak Angle:** {avg_height:.1f}°\n"

        # Dummy Recommendations
        report += "\n**Recommendations: (Dummy Recommendations)**\n"
        depth_feedback = "Go Lower" if exercise_type == "pushup" else "Go Deeper"
        height_feedback = "Extend Arms Fully" if exercise_type == "pushup" else "Stand Straighter"
        angle_down_threshold = thresholds[exercise_type]["down"]
        angle_up_threshold = thresholds[exercise_type]["up"]

        if common_corrections.get(depth_feedback, 0) > total_reps * 0.3: # If depth correction needed often
            report += f"*   **Focus on Depth:** Try to consistently reach a lower position (around {angle_down_threshold}° or less) in your {exercise_type}s.\n"
        elif avg_depth is not None and avg_depth > angle_down_threshold + 10:
             report += f"*   **Improve Depth:** Your average depth was {avg_depth:.1f}°. Aim for closer to {angle_down_threshold}°.\n"

        if common_corrections.get(height_feedback, 0) > total_reps * 0.2:
             report += f"*   **Focus on Peak:** Ensure you reach the full top position (around {angle_up_threshold}°) in your {exercise_type}s.\n"
        elif avg_height is not None and avg_height < angle_up_threshold - 10:
             report += f"*   **Improve Peak:** Your average peak angle was {avg_height:.1f}°. Aim for closer to {angle_up_threshold}°.\n"

        # Asymmetry Check (Simple Example)
        left_angles_down = [d['left_angle'] for d in collected_data if d['state'] == 'down' and d['left_angle'] is not None]
        right_angles_down = [d['right_angle'] for d in collected_data if d['state'] == 'down' and d['right_angle'] is not None]
        if left_angles_down and right_angles_down:
             avg_diff = np.mean(np.abs(np.array(left_angles_down) - np.array(right_angles_down)))
             if avg_diff > 10: # Threshold for significant difference
                 report += f"*   **Check Symmetry:** There was an average difference of {avg_diff:.1f}° between your left and right side angles during the down phase. Focus on balanced movement.\n"


    elif exercise_type == "plank":
        hold_periods = [d['hold_duration'] for d in collected_data if d['state'] == 'hold']
        max_hold = max(hold_periods) if hold_periods else 0
        report += f"*   **Longest Continuous Hold (Correct Form):** {max_hold:.1f} seconds\n"

        # Dummy Recommendations
        report += "\n**Recommendations:**\n"
        if common_corrections.get("Lift Hips", 0) > 5: # Arbitrary count threshold
            report += "*   **Engage Core:** Focus on keeping your hips aligned with your shoulders and ankles. Avoid letting them sag.\n"
        if common_corrections.get("Lower Hips", 0) > 5:
            report += "*   **Body Alignment:** Ensure your hips aren't too high. Aim for a straight line from shoulders to ankles.\n"
        if max_hold < 15 and total_duration > 15: # If they held less than 15s but tried longer
             report += "*   **Build Endurance:** Work on maintaining the correct plank form for longer durations.\n"


    # General Feedback Summary
    if common_corrections:
        report += "\n*   **Common Corrections During Session:**\n"
        for fb, count in common_corrections.items():
            report += f"    *   {fb}: {count} times\n"
    else:
         report += "\n*   **Form Consistency:** Good job maintaining form! No major corrections were frequently triggered.\n"

    if feedback_counter["No person detected"] > 10 or feedback_counter["Landmarks missing"] > 10:
         report += "\n*   **Visibility Note:** The system had trouble detecting the pose at times. Ensure good lighting and full body visibility in the frame for best results.\n"

    return report


# --- Streamlit App Interface ---
st.set_page_config(layout="wide")

# --- Sidebar ---
st.sidebar.title("Settings")
st.sidebar.write("Select an exercise and upload your video.")
exercise_options = ["Pushup", "Squat", "Plank"]
selected_exercise = st.sidebar.selectbox(
    "Choose an Exercise:", options=exercise_options, index=None, placeholder="Select exercise..."
)
uploaded_file = None
if selected_exercise:
    uploaded_file = st.sidebar.file_uploader(
        f"Upload a video for {selected_exercise} analysis:", type=["mp4", "mov", "avi", "mkv"]
    )
else:
    st.sidebar.info("Please select an exercise first.")

# --- Main Area ---
st.title("🏋️ AI Workout Form Analyzer")
video_placeholder = st.empty()
processing_message = st.empty()
report_placeholder = st.empty() # Placeholder for the final report

# --- Processing Logic ---
if uploaded_file is not None and selected_exercise:
    processing_message.info("Processing video... Please wait.")
    video_placeholder.empty()
    report_placeholder.empty()

    temp_video_path = None
    all_frame_data = [] # List to collect metrics from all frames

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

        # Process video frame by frame, display frame, collect metrics
        frames_processed = False
        for processed_frame, frame_metrics in process_video(temp_video_path, selected_exercise.lower()):
            if processed_frame is not None:
                video_placeholder.image(processed_frame, channels="RGB")
                all_frame_data.append(frame_metrics) # Collect metrics
                frames_processed = True
            else:
                # Handle case where process_video failed early
                if not frames_processed: # Only show error if no frames were ever processed
                    processing_message.error("Failed to open or process video file.")
                break # Stop processing

        # --- Generate and Display Report AFTER loop ---
        if frames_processed:
            processing_message.success("Processing complete! Generating report...")
            final_report = generate_workout_report(all_frame_data, selected_exercise.lower())
            report_placeholder.markdown(final_report)
            processing_message.success("Processing complete! Report generated below.") # Update message
        elif temp_video_path and not all_frame_data: # File existed but no data collected (e.g., empty video)
             processing_message.warning("Video processed, but no pose data was detected to generate a report.")

    except Exception as e:
        processing_message.error(f"An error occurred during processing: {e}")
        st.exception(e) # Show full traceback for debugging
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
             try:
                 os.unlink(temp_video_path)
             except Exception as e:
                 st.warning(f"Could not delete temporary file {temp_video_path}: {e}")

elif selected_exercise:
    video_placeholder.info("Please upload a video file using the sidebar to begin analysis.")
else:
    video_placeholder.info("Select an exercise from the sidebar to get started.")