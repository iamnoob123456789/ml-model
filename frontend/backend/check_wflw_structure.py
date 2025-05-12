import os

# Define paths
data_dir = r"C:\Users\shrey\Downloads"
annotations_dir = os.path.join(data_dir, "WFLW_annotations")

# Check contents of WFLW_annotations
if os.path.exists(annotations_dir):
    print("Contents of WFLW_annotations:")
    for root, dirs, files in os.walk(annotations_dir):
        level = root.replace(annotations_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}    {f}")
else:
    print(f"Annotations directory not found: {annotations_dir}")