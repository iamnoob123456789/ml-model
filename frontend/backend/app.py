import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import torch
import mediapipe as mp
from datetime import datetime
from supervision import ByteTrack, Detections

# Set environment variable to handle OMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all endpoints

# Set absolute paths for directories
backend_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(backend_dir, 'Uploads')
app.config['SNAPSHOTS_FOLDER'] = os.path.join(backend_dir, 'snapshots')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SNAPSHOTS_FOLDER'], exist_ok=True)

# Save the original torch.load function
original_load = torch.load

# Define a custom torch.load that sets weights_only=False
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

# Temporarily override torch.load
torch.load = custom_load

# Load the YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Restore the original torch.load
torch.load = original_load

# Load pre-trained models
tracker = ByteTrack()  # ByteTrack for tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=50, min_detection_confidence=0.5)

# Frame skipping for optimization
FRAME_SKIP = 10  # Process every 10th frame

def estimate_head_pose(landmarks, frame_shape):
    """Estimate head pose based on facial landmarks and classify attentiveness."""
    nose = landmarks.landmark[1]  # Nose tip
    left_eye = landmarks.landmark[33]  # Left eye corner
    right_eye = landmarks.landmark[263]  # Right eye corner

    eye_center_x = (left_eye.x + right_eye.x) / 2
    yaw = abs(eye_center_x - nose.x) * frame_shape[1]

    if yaw < 50:  # If head is facing forward (within 50 pixels)
        return "attentive"
    return "inattentive"

def process_video(video_path):
    """Process video to detect and track students, and calculate attentiveness scores."""
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        raise ValueError("Could not open video file")
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    student_data = {}  # {tracker_id: {"status": [], "snapshot": path}}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_count}")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        print(f"Processing frame {frame_count}...")
        try:
            # Detect people using YOLOv8
            results = yolo_model(frame)
            print(f"YOLO detected {len(results[0].boxes.xyxy)} objects")
            detections = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

            if len(detections) > 0:
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
                detections_sup = Detections(xyxy=detections, confidence=confidences, class_id=class_ids)
                print(f"Created Detections object with {len(detections)} detections")

                # Update tracker with detections
                tracks = tracker.update_with_detections(detections_sup)
                print(f"Tracker returned {len(tracks)} tracks")

                # Process each tracked object
                for track in tracks:
                    try:
                        # Extract track information (track is a tuple: [x1, y1, x2, y2], None, confidence, class_id, track_id, ...)
                        if len(track) < 5:
                            print(f"Skipping track with insufficient elements: {track}")
                            continue

                        # Extract bounding box coordinates (first element is a 4-element array)
                        bbox = track[0]
                        if not isinstance(bbox, np.ndarray) or bbox.size != 4:
                            print(f"Skipping track with invalid bounding box: {track}")
                            continue

                        x1, y1, x2, y2 = map(int, bbox)  # Convert the 4-element array to integers
                        track_id = int(track[4])  # Extract track_id from index 4

                        # Validate bounding box coordinates
                        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                            print(f"Invalid bounding box for track {track_id}: ({x1}, {y1}, {x2}, {y2})")
                            continue

                        # Extract face region
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size == 0 or face_img is None:
                            print(f"Skipping save: Invalid face_img for track_id {track_id}")
                            continue

                        # Convert to RGB for MediaPipe
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(face_rgb)
                        print(f"Face mesh processed for track {track_id}")

                        if results.multi_face_landmarks:
                            for landmarks in results.multi_face_landmarks:
                                if not landmarks.landmark:
                                    continue
                                status = estimate_head_pose(landmarks, frame.shape)

                                # Initialize student data if not present
                                if track_id not in student_data:
                                    snapshot_path = os.path.join(app.config['SNAPSHOTS_FOLDER'], f"student_{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                                    success = cv2.imwrite(snapshot_path, face_img)
                                    print(f"Image saved: {snapshot_path}, Success: {success}, Path exists: {os.path.exists(snapshot_path)}")
                                    if not success:
                                        print(f"Failed to save image for track_id {track_id}")
                                        continue
                                    snapshot_url = f"/snapshots/{os.path.basename(snapshot_path)}"
                                    student_data[track_id] = {"status": [], "snapshot": snapshot_url}

                                student_data[track_id]["status"].append(status)
                    except Exception as e:
                        print(f"Error processing track in frame {frame_count}: {e}, Track: {track}")
                        continue
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()
    print(f"Video processing completed. Processed {frame_count} frames")

    # Calculate attentiveness scores
    response_data = []
    total_students = len(student_data)
    attentive_count = 0

    for track_id, data in student_data.items():
        status_list = data["status"]
        attentive_frames = status_list.count("attentive")
        total_frames = len(status_list)
        score = (attentive_frames / total_frames) * 100 if total_frames > 0 else 0

        overall_status = "attentive" if score > 70 else "inattentive"
        if overall_status == "attentive":
            attentive_count += 1

        response_data.append({
            "student_id": track_id,
            "score": round(score, 2),
            "status": overall_status,
            "snapshot": data["snapshot"]
        })

    inattentive_count = total_students - attentive_count
    attentive_percentage = (attentive_count / total_students * 100) if total_students > 0 else 0
    inattentive_percentage = (inattentive_count / total_students * 100) if total_students > 0 else 0

    print(f"Completed processing. Total students: {total_students}")
    return {
        "students": response_data,
        "total_students": total_students,
        "attentive_percentage": round(attentive_percentage, 2),
        "inattentive_percentage": round(inattentive_percentage, 2)
    }

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    """Serve snapshot images."""
    print(f"Serving snapshot: {filename}")
    return send_from_directory(app.config['SNAPSHOTS_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    print("Received upload request")
    if 'video' not in request.files:
        print("No video file in request")
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    print(f"Video saved to {video_path}")

    try:
        result = process_video(video_path)
        print("Upload processing completed")
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full stack trace
        print(f"Error during upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)