import os

# Define directories
train_image_dir = "datasets/all/train/images_transformed"
train_target_dir = "datasets/all/train/annotations_transformed"
val_image_dir = "datasets/all/val/images_transformed"
val_target_dir = "datasets/all/val/annotations_transformed"

# Function to count valid files in a directory
def count_valid_files(directory, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = [".jpg", ".jpeg", ".png", ".xml", ".txt"]  # Default valid extensions
    
    return sum(1 for file in os.listdir(directory) if file.lower().endswith(tuple(valid_extensions)))

# Count valid files in each directory
train_image_count = count_valid_files(train_image_dir, valid_extensions=[".jpg", ".jpeg", ".png"])
train_target_count = count_valid_files(train_target_dir, valid_extensions=[".xml", ".txt"])
val_image_count = count_valid_files(val_image_dir, valid_extensions=[".jpg", ".jpeg", ".png"])
val_target_count = count_valid_files(val_target_dir, valid_extensions=[".xml", ".txt"])

# Print the results
print(f"Train Images: {train_image_count}")
print(f"Train Annotations: {train_target_count}")
print(f"Validation Images: {val_image_count}")
print(f"Validation Annotations: {val_target_count}")
