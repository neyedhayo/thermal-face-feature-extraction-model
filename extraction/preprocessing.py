from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def normalize_image(image):
    return np.array(image) / 255.0

class ImagePreprocessor:
    def __init__(self, file_path, processed_path):
        self.file_path = file_path
        self.processed_path = processed_path

        # Create directories for processed images
        os.makedirs(os.path.join(processed_path, 'train', 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(processed_path, 'train', 'thermal'), exist_ok=True)
        os.makedirs(os.path.join(processed_path, 'test', 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(processed_path, 'test', 'thermal'), exist_ok=True)
        os.makedirs(os.path.join(processed_path, 'validate', 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(processed_path, 'validate', 'thermal'), exist_ok=True)

    def process_images(self, file_list, image_type, phase):
        target_size = (224, 224)  # Desired output size of the images
        target_dir = os.path.join(self.processed_path, phase, image_type)

        for filename in tqdm(file_list, desc=f'Processing {image_type} images for {phase}'):
            try:
                with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                    with zip_ref.open(filename) as file:
                        image = Image.open(file).convert('RGB')

                # Resize and normalize the image
                image_resized = np.array(image.resize(target_size))
                image_normalized = normalize_image(image_resized)

                target_path = os.path.join(target_dir, os.path.basename(filename))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                cv2.imwrite(target_path, (image_normalized * 255).astype(np.uint8))

            except Exception as e:
                print(f"Error processing image {filename}: {e}")

    def split_and_process(self):
        image_types = ['RGB-faces-128x128', 'thermal-face-128x128']
        for image_type in image_types:
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                images = [f for f in zip_ref.namelist() if image_type in f and f.endswith('.jpg')]

                train_test_temp, validate_files = train_test_split(images, test_size=510, random_state=42)
                train_files, test_files = train_test_split(train_test_temp, test_size=510, train_size=512, random_state=42)

                self.process_images(train_files, image_type.split('-')[0].lower(), 'train')
                self.process_images(test_files, image_type.split('-')[0].lower(), 'test')
                self.process_images(validate_files, image_type.split('-')[0].lower(), 'validate')

def main():
    file_path = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/raw/face Dataset.zip'
    processed_path = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/normalized'
    preprocessor = ImagePreprocessor(file_path, processed_path)
    preprocessor.split_and_process()
    print("Data preprocessing complete.")

if __name__ == "__main__":
    main()



