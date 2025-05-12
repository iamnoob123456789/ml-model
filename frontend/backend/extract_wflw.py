import tarfile
import os

# Define paths
data_dir = r"C:/Users/shrey/Downloads"
images_tar = os.path.join(data_dir, "WFLW_images.tar.gz")
annotations_tar = os.path.join(data_dir, "WFLW_annotations.tar.gz")

# Extract images
if os.path.exists(images_tar):
    with tarfile.open(images_tar, "r:gz") as tar:
        tar.extractall(path=os.path.join(data_dir, "WFLW_images"), filter='data')
    print("Images extracted successfully.")
else:
    print(f"Image file not found: {images_tar}")

# Extract annotations
if os.path.exists(annotations_tar):
    with tarfile.open(annotations_tar, "r:gz") as tar:
        tar.extractall(path=os.path.join(data_dir, "WFLW_annotations"), filter='data')
    print("Annotations extracted successfully.")
else:
    print(f"Annotations file not found: {annotations_tar}")