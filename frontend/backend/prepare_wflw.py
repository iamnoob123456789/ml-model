import os
import cv2
import numpy as np
import shutil
import random

# Define paths
dataset_dir = r"C:\Users\shrey\Downloads"
output_dir = r"C:\Users\shrey\Downloads\dataset"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# WFLW keypoint indices for nose, left eye, right eye, left mouth, right mouth (from 98 landmarks)
keypoint_indices = [60, 76, 82, 87, 93]  # Selected indices for 5 keypoints

# Create directories for train, val, and test splits
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Load WFLW annotations
anno_file_train = os.path.join(dataset_dir, "WFLW_annotations", "WFLW_annotations", "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_train.txt")
anno_file_test = os.path.join(dataset_dir, "WFLW_annotations", "WFLW_annotations", "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_test.txt")
image_dir = os.path.join(dataset_dir, "WFLW_images", "WFLW_images")  # Adjusted for nested WFLW_images directory

print(f"Checking annotation files: train={os.path.exists(anno_file_train)}, test={os.path.exists(anno_file_test)}")
print(f"Image directory exists: {os.path.exists(image_dir)}")

# Process training and validation data
train_lines = []
val_lines = []
if os.path.exists(anno_file_train):
    with open(anno_file_train, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)  # Shuffle to split into train and val
        val_split = int(0.2 * len(lines))  # 20% for validation
        train_lines = lines[val_split:]
        val_lines = lines[:val_split]
    print(f"Train lines: {len(train_lines)}, Val lines: {len(val_lines)}")
else:
    raise FileNotFoundError(f"Train annotation file not found at {anno_file_train}")

# Process test data
test_lines = []
if os.path.exists(anno_file_test):
    with open(anno_file_test, 'r') as f:
        test_lines = f.readlines()
    print(f"Test lines: {len(test_lines)}")
else:
    raise FileNotFoundError(f"Test annotation file not found at {anno_file_test}")

# Process and split data
total_processed = 0
missing_images = 0
skipped_invalid = 0
for split, lines in [('train', train_lines), ('val', val_lines), ('test', test_lines)]:
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 196:
            print(f"Skipping malformed line: {line}")
            continue
        keypoints = list(map(float, parts[:196]))
        img_file = parts[-1]
        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            missing_images += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            missing_images += 1
            continue
        h, w = img.shape[:2]

        # Select 5 keypoints
        selected_kps = []
        for i in keypoint_indices:
            x = keypoints[i * 2]
            y = keypoints[i * 2 + 1]
            selected_kps.append([x, y])

        # Compute bounding box from keypoints
        x_coords = [kp[0] for kp in selected_kps]
        y_coords = [kp[1] for kp in selected_kps]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding to the bounding box (10% of the box size)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)

        # Check for invalid bounding box (zero width or height)
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h
        if box_width <= 0 or box_height <= 0:
            print(f"Skipping invalid bounding box for {img_file}: width={box_width}, height={box_height}")
            skipped_invalid += 1
            continue

        # Compute normalized bounding box coordinates
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h

        # Normalize keypoints and add visibility flags
        normalized_kps = []
        for x, y in selected_kps:
            x_norm, y_norm = x / w, y / h
            visibility = 1  # Assume all keypoints are visible
            normalized_kps.extend([x_norm, y_norm, visibility])

        # Write to YOLO format (class_id + bbox + keypoints)
        label_path = os.path.join(output_dir, "labels", split, img_file.replace('/', '_').replace('.jpg', '.txt'))
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {box_width} {box_height} {' '.join(map(str, normalized_kps))}\n")

        # Copy image
        shutil.copy(img_path, os.path.join(output_dir, "images", split, img_file.replace('/', '_')))
        total_processed += 1

print(f"Total images processed and split: {total_processed}")
print(f"Total missing or corrupted images: {missing_images}")
print(f"Total images skipped due to invalid bounding boxes: {skipped_invalid}")

# Create data.yaml
with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
    f.write(f"""
train: {os.path.join(output_dir, "images/train").replace("\\", "/")}
val: {os.path.join(output_dir, "images/val").replace("\\", "/")}
test: {os.path.join(output_dir, "images/test").replace("\\", "/")}

nc: 1
names: ['person']
kpt_shape: [5, 3]  # 5 keypoints with (x, y, visibility)
""")