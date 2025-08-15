import os
from PIL import Image, UnidentifiedImageError

IMAGE_DIR = "jpg"
ANNOTATION_DIR = "xml"

def get_filenames_without_extension(directory, extensions=None):
    """Get filenames without extension from a directory."""
    filenames = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            name, ext = os.path.splitext(filename)
            if extensions is None or ext.lower() in extensions:
                filenames.add(name)
    return filenames

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")

def verify_image(file_path):
    """Check if an image file is valid and complete."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False
    except Exception:
        return False

# Step 1: Find unmatched files
image_filenames = get_filenames_without_extension(IMAGE_DIR, {".jpg", ".jpeg", ".png"})
annotation_filenames = get_filenames_without_extension(ANNOTATION_DIR, {".xml"})

unmatched_images = image_filenames - annotation_filenames
unmatched_annotations = annotation_filenames - image_filenames

for name in unmatched_images:
    delete_file(os.path.join(IMAGE_DIR, name + ".jpg"))
    delete_file(os.path.join(IMAGE_DIR, name + ".jpeg"))
    delete_file(os.path.join(IMAGE_DIR, name + ".png"))

for name in unmatched_annotations:
    delete_file(os.path.join(ANNOTATION_DIR, name + ".xml"))

# Step 2: Verify images & remove corrupted ones and their annotations
for filename in os.listdir(IMAGE_DIR):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    xml_path = os.path.join(ANNOTATION_DIR, name + ".xml")

    if not verify_image(img_path):
        print(f"Corrupted image detected: {img_path}")
        delete_file(img_path)
        delete_file(xml_path)  # also delete annotation

print("Dataset cleanup complete.")
