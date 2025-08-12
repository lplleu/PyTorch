import os

# Folders with images and annotations
annots_dir = 'annots'
imgs_dir = 'imgs'

def is_valid_bbox(xmin, ymin, xmax, ymax):
    return (xmax > xmin) and (ymax > ymin)

def parse_annotation(file_path):
    """
    Example for VOC format (xmin, ymin, xmax, ymax) per line or
    YOLO format (class, x_center, y_center, width, height)
    Adjust parsing logic if needed.
    """
    boxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            # Assuming YOLO format: class x_center y_center width height (all normalized 0-1)
            cls, xc, yc, w, h = parts
            xc, yc, w, h = map(float, (xc, yc, w, h))
            if w <= 0 or h <= 0:
                return False
        elif len(parts) == 4:
            # Assuming VOC format: xmin ymin xmax ymax
            xmin, ymin, xmax, ymax = map(float, parts)
            if not is_valid_bbox(xmin, ymin, xmax, ymax):
                return False
        else:
            # Unknown format - skip or treat as invalid
            return False
    return True

def cleanup_invalid_boxes(annots_dir, imgs_dir):
    annots_files = os.listdir(annots_dir)
    removed_files = []
    for annot_file in annots_files:
        annot_path = os.path.join(annots_dir, annot_file)
        if not parse_annotation(annot_path):
            # Delete annotation file
            os.remove(annot_path)
            # Delete corresponding image
            # Assumes image file has the same base name with common extensions
            base_name = os.path.splitext(annot_file)[0]
            deleted_image = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_path = os.path.join(imgs_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_image = True
                    break
            removed_files.append((annot_file, deleted_image))
            print(f"Removed annotation {annot_file} and image: {deleted_image}")
    print(f"Total removed: {len(removed_files)}")

cleanup_invalid_boxes(annots_dir, imgs_dir)
