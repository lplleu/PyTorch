import os
import shutil
import xml.etree.ElementTree as ET

# Paths
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
IMAGE_DIR_OUT = "images2"
ANNOTATION_DIR_OUT = "annotations2"

# Minimum size for a bounding box (width and height)
MIN_WIDTH = 20
MIN_HEIGHT = 20

# Create output directories if they don't exist
os.makedirs(IMAGE_DIR_OUT, exist_ok=True)
os.makedirs(ANNOTATION_DIR_OUT, exist_ok=True)

def has_valid_objects(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            width = xmax - xmin
            height = ymax - ymin
            if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                return True
    except Exception as e:
        print(f"Failed to parse {xml_path}: {e}")
    return False

# Go through annotation files
for fname in os.listdir(ANNOTATION_DIR):
    if not fname.endswith(".xml"):
        continue
    ann_path = os.path.join(ANNOTATION_DIR, fname)
    if has_valid_objects(ann_path):
        # Copy annotation
        shutil.copy2(ann_path, os.path.join(ANNOTATION_DIR_OUT, fname))
        
        # Copy corresponding image
        base_name = os.path.splitext(fname)[0]
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(IMAGE_DIR_OUT, base_name + ext))
                break
        else:
            print(f"No image found for annotation {fname}")

print("Done filtering and copying valid files.")
