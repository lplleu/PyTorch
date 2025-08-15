import os
import xml.etree.ElementTree as ET

# Path to your dataset
dataset_dir = "/train" 

images_dir = os.path.join(dataset_dir, "images")
annotations_dir = os.path.join(dataset_dir, "annotations")

deleted_files = []

for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(annotations_dir, xml_file)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        invalid = False
        for obj in root.findall(".//object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            width = xmax - xmin
            height = ymax - ymin

            if width <= 0 or height <= 0:
                invalid = True
                break

        if invalid:
            # Remove XML
            os.remove(xml_path)

            # Remove corresponding JPG
            jpg_filename = os.path.splitext(xml_file)[0] + ".jpg"
            jpg_path = os.path.join(images_dir, jpg_filename)
            if os.path.exists(jpg_path):
                os.remove(jpg_path)

            deleted_files.append(jpg_filename)

    except ET.ParseError:
        # Malformed XML — also delete
        os.remove(xml_path)
        jpg_filename = os.path.splitext(xml_file)[0] + ".jpg"
        jpg_path = os.path.join(images_dir, jpg_filename)
        if os.path.exists(jpg_path):
            os.remove(jpg_path)
        deleted_files.append(jpg_filename)

print(f"Deleted {len(deleted_files)} files with invalid bounding boxes.")
