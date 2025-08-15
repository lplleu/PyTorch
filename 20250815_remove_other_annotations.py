import os
import xml.etree.ElementTree as ET

ANNOTATION_DIR = "xml"  
IMAGE_DIR = "jpg"        
TARGET_LABEL = "car"    

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")

for filename in os.listdir(ANNOTATION_DIR):
    if not filename.lower().endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find and remove all 'car' objects
    removed = False
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is not None and name_tag.text.strip().lower() == TARGET_LABEL.lower():
            root.remove(obj)
            removed = True

    if removed:
        # Check if any objects remain
        remaining_objects = root.findall("object")
        if len(remaining_objects) == 0:
            # Delete annotation and corresponding image
            delete_file(xml_path)
            jpg_path = os.path.join(IMAGE_DIR, os.path.splitext(filename)[0] + ".jpg")
            delete_file(jpg_path)
        else:
            # Save modified XML
            tree.write(xml_path)
            print(f"Updated XML (removed '{TARGET_LABEL}'): {xml_path}")

print("object removal complete.")
