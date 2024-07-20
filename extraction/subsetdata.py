import os
import shutil
import random

def create_subset(source_dir, dest_dir, total_subset_size):
    """
    Copies a subset of images from the source directory to the destination directory.

    Parameters:
    - source_dir: The source directory containing the original images.
    - dest_dir: The destination directory where the subset will be stored.
    - total_subset_size: The total number of images to copy for each subset.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    classes = os.listdir(source_dir)
    subset_size_per_class = total_subset_size // len(classes)
    remaining_images = total_subset_size % len(classes)

    for class_name in classes:
        class_source_dir = os.path.join(source_dir, class_name)
        class_dest_dir = os.path.join(dest_dir, class_name)

        if not os.path.exists(class_dest_dir):
            os.makedirs(class_dest_dir)

        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) < subset_size_per_class + (1 if remaining_images > 0 else 0):
            subset_files = image_files
        else:
            subset_files = random.sample(image_files, subset_size_per_class + (1 if remaining_images > 0 else 0))
            if remaining_images > 0:
                remaining_images -= 1

        # Copy each file to the destination directory
        for file_name in subset_files:
            src_file = os.path.join(class_source_dir, file_name)
            dest_file = os.path.join(class_dest_dir, file_name)
            shutil.copy(src_file, dest_file)

train_source_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase/train'
val_source_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase/val'
test_source_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase/test'

train_dest_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/train'
val_dest_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/val'
test_dest_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/test'

train_total_subset_size = 1168
val_total_subset_size = 1183
test_total_subset_size = 1184

directories = [train_source_dir, val_source_dir, test_source_dir]

for directory in directories:
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
    else:
        print(f"Directory exists: {directory}")

# Create subsets
create_subset(train_source_dir, train_dest_dir, train_total_subset_size)
create_subset(val_source_dir, val_dest_dir, val_total_subset_size)
create_subset(test_source_dir, test_dest_dir, test_total_subset_size)

print("Subset creation complete.")

