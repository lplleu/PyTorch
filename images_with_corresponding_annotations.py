import os
import shutil

# Directories
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
DEST_DIR = "images3"

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Get list of annotation filenames (without extension)
annotation_files = {os.path.splitext(f)[0] for f in os.listdir(ANNOTATION_DIR) if f.endswith(".xml")}

# Loop through images and copy if annotation exists
for image_file in os.listdir(IMAGE_DIR):
    name, ext = os.path.splitext(image_file)
    if name in annotation_files:
        src = os.path.join(IMAGE_DIR, image_file)
        dst = os.path.join(DEST_DIR, image_file)
        shutil.copy2(src, dst)

print(f"Copied images with annotations to {DEST_DIR}")
