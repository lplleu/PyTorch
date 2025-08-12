import os
import shutil
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image

# =========================
# CONFIGURATION
# =========================
images_dir = "images"
labels_dir = "labels"
output_xml_dir = "annotations"
class_names = ["class0", "class1", "class2"]
min_object_area_ratio = 0.001  # Minimum bbox area (fraction of image area) to keep

os.makedirs(output_xml_dir, exist_ok=True)

# =========================
# STEP 1: Remove empty YOLO annotations
# =========================
def remove_empty_annotations():
    removed = 0
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            img_file = os.path.splitext(label_file)[0] + ".jpg"
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                os.remove(img_path)
            os.remove(label_path)
            removed += 1
    print(f"[Step 1] Removed {removed} empty annotations & images.")

# =========================
# STEP 2: Convert YOLO to Pascal VOC
# =========================
def yolo_to_voc():
    converted = 0
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        img_name = os.path.splitext(label_file)[0]
        img_path = os.path.join(images_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        w, h = img.size

        annotation = Element("annotation")
        SubElement(annotation, "folder").text = os.path.basename(images_dir)
        SubElement(annotation, "filename").text = os.path.basename(img_path)

        size = SubElement(annotation, "size")
        SubElement(size, "width").text = str(w)
        SubElement(size, "height").text = str(h)
        SubElement(size, "depth").text = str(len(img.getbands()))

        with open(os.path.join(labels_dir, label_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            cls_id, x_center, y_center, bw, bh = map(float, line.strip().split())
            cls_id = int(cls_id)
            xmin = int((x_center - bw / 2) * w)
            xmax = int((x_center + bw / 2) * w)
            ymin = int((y_center - bh / 2) * h)
            ymax = int((y_center + bh / 2) * h)

            obj = SubElement(annotation, "object")
            SubElement(obj, "name").text = class_names[cls_id]
            SubElement(obj, "pose").text = "Unspecified"
            SubElement(obj, "truncated").text = "0"
            SubElement(obj, "difficult").text = "0"

            bndbox = SubElement(obj, "bndbox")
            SubElement(bndbox, "xmin").text = str(xmin)
            SubElement(bndbox, "ymin").text = str(ymin)
            SubElement(bndbox, "xmax").text = str(xmax)
            SubElement(bndbox, "ymax").text = str(ymax)

        xml_str = parseString(tostring(annotation)).toprettyxml(indent="  ")
        with open(os.path.join(output_xml_dir, img_name + ".xml"), "w") as f:
            f.write(xml_str)
        converted += 1

    print(f"[Step 2] Converted {converted} YOLO annotations to Pascal VOC XML.")

# =========================
# STEP 3: Validate Annotations
# =========================
def validate_annotations():
    removed = 0
    for xml_file in os.listdir(output_xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(output_xml_dir, xml_file)

        with open(xml_path, "r") as f:
            content = f.read()

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_w = int(root.find("size/width").text)
            img_h = int(root.find("size/height").text)
            image_area = img_w * img_h

            valid = False
            for obj in root.findall("object"):
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)
                bbox_area = (xmax - xmin) * (ymax - ymin)
                if bbox_area / image_area >= min_object_area_ratio:
                    valid = True
                    break

            if not valid:
                img_file = os.path.splitext(xml_file)[0] + ".jpg"
                img_path = os.path.join(images_dir, img_file)
                if os.path.exists(img_path):
                    os.remove(img_path)
                os.remove(xml_path)
                removed += 1

        except Exception as e:
            print(f"Error reading {xml_file}: {e}")

    print(f"[Step 3] Removed {removed} invalid annotations & images.")

# =========================
# STEP 4: Count total images and per class
# =========================
def count_stats():
    total_images = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    print(f"[Step 4] Total images: {total_images}")

    from collections import Counter
    class_counts = Counter()

    for xml_file in os.listdir(output_xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        import xml.etree.ElementTree as ET
        tree = ET.parse(os.path.join(output_xml_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall("object"):
            class_counts[obj.find("name").text] += 1

    print("[Step 4] Annotations per class:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")

# =========================
# RUN PIPELINE
# =========================
remove_empty_annotations()
yolo_to_voc()
validate_annotations()
count_stats()
