import os
import shutil
import xml.etree.ElementTree as ET

# input folders
imgs_dir = "imgs"
xmls_dir = "xmls"

# output folders
imgs0_dir = "imgs0"
xmls0_dir = "xmls0"

# make output dirs if they don't exist
os.makedirs(imgs0_dir, exist_ok=True)
os.makedirs(xmls0_dir, exist_ok=True)

# go through all xml files
for xml_file in os.listdir(xmls_dir):
    if not xml_file.endswith(".xml"):
        continue
    
    xml_path = os.path.join(xmls_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # check if any object name is 'house'
    has_house = any(
        obj.find("name") is not None and obj.find("name").text == "house"
        for obj in root.findall("object")
    )

    if has_house:
        # copy xml
        shutil.copy2(xml_path, os.path.join(xmls0_dir, xml_file))
        
        # copy corresponding image
        base_name = os.path.splitext(xml_file)[0]
        # check common extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            img_file = base_name + ext
            img_path = os.path.join(imgs_dir, img_file)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(imgs0_dir, img_file))
                break
