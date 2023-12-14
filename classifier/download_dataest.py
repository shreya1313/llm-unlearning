import os
import subprocess
from zipfile import ZipFile

def extract_zip_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                with ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_file_path)

# Step 1: Create a new directory named 'data'
data_directory = 'data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Step 2: Download the dataset into the 'data' directory
download_command = 'kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p {}'.format(data_directory)
subprocess.run(download_command, shell=True)

# Step 3: Find the downloaded zip file
zip_file = [f for f in os.listdir(data_directory) if f.endswith('.zip')][0]
zip_file_path = os.path.join(data_directory, zip_file)

# Step 4: Extract the contents of the zip file into the 'data' directory
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_directory)

# Step 5: Remove the downloaded zip file
os.remove(zip_file_path)

# Step 6: Recursively extract any remaining zip files in the 'data' directory
extract_zip_files(data_directory)

print("Dataset downloaded and all zip files extracted successfully in the 'data' directory.")
