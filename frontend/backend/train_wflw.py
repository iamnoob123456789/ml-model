from ultralytics import YOLO

# Load the pretrained YOLOv8-Pose model
pose_model = YOLO('yolov8n-pose.pt')

# Train the model
results = pose_model.train(
    data=r"C:\Users\shrey\Downloads\dataset\data.yaml",
    imgsz=640,
    epochs=15,
    batch=16,
    name='custom_pose_wflw',
    pretrained=True,
    device='cpu'  # Use 'cuda' or '0' if you have a compatible GPU
)

# Save the trained model
pose_model.save(r"C:\Users\shrey\Desktop\Project-Vikram\AttendEase\frontend\backend\custom_pose_wflw.pt")