import zipfile
import os

zip_path = 'data/raw/face Dataset.zip'
extraction_path = '/data/processed'

os.makedirs(extraction_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

print("Extraction complete.")

