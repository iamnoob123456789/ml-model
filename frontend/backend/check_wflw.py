import os

# Define paths
data_dir = r"C:\Users\shrey\Downloads"
images_dir = os.path.join(data_dir, "WFLW_images")

# Check contents of WFLW_images
if os.path.exists(images_dir):
    print("Contents of WFLW_images:")
    for root, dirs, files in os.walk(images_dir):
        level = root.replace(images_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}    {f}")
else:
    print(f"Images directory not found: {images_dir}")