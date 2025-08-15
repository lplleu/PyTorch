import os

IMAGE_DIR = os.path.join(os.getcwd(), "jpg")  # safer path
ANNOTATION_DIR = os.path.join(os.getcwd(), "xml")

def get_filenames_without_extension(directory, extensions=None):
    """Gets a set of filenames without extensions from a directory."""
    filenames = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            name, ext = os.path.splitext(filename)
            if extensions is None or ext.lower() in extensions:
                filenames.add(name.lower())  # normalize to lowercase
    return filenames

image_filenames = get_filenames_without_extension(IMAGE_DIR, {".jpg", ".jpeg", ".png"})
annotation_filenames = get_filenames_without_extension(ANNOTATION_DIR, {".xml"})

# Find unmatched files
unmatched_images = image_filenames - annotation_filenames
unmatched_annotations = annotation_filenames - image_filenames

def delete_unmatched_files(directory, unmatched_filenames):
    """Deletes files in a directory that are in the set of unmatched filenames."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            name, ext = os.path.splitext(filename)
            if name.lower() in unmatched_filenames:
                file_path = os.path.join(directory, filename)
                print(f"Deleting unmatched file: {file_path}")
                # os.remove(file_path)  # Uncomment to actually delete

print(f"Unmatched images: {unmatched_images}")
print(f"Unmatched annotations: {unmatched_annotations}")

# Uncomment to delete
# delete_unmatched_files(IMAGE_DIR, unmatched_images)
# delete_unmatched_files(ANNOTATION_DIR, unmatched_annotations)

print("Unmatched file check complete.")
