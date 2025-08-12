import os
import xml.etree.ElementTree as ET

annots_dir = 'annots'
imgs_dir = 'imgs'

def is_valid_bbox(xmin, ymin, xmax, ymax):
    return xmax > xmin and ymax > ymin

def annotation_has_invalid_bbox(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if not is_valid_bbox(xmin, ymin, xmax, ymax):
                print(f"Invalid bbox in {xml_path}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                return True
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return True  # treat parse error as invalid
    return False

def delete_invalid_annotations(annots_dir, imgs_dir):
    annots_files = os.listdir(annots_dir)
    removed_count = 0
    for annot_file in annots_files:
        if not annot_file.endswith('.xml'):
            continue
        annot_path = os.path.join(annots_dir, annot_file)
        if annotation_has_invalid_bbox(annot_path):
            # Delete annotation
            os.remove(annot_path)
            # Delete corresponding image (same base name with common extensions)
            base_name = os.path.splitext(annot_file)[0]
            deleted_image = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_path = os.path.join(imgs_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_image = True
                    break
            print(f"Deleted annotation {annot_file} and image: {deleted_image}")
            removed_count += 1
    print(f"Total invalid annotations removed: {removed_count}")

delete_invalid_annotations(annots_dir, imgs_dir)
