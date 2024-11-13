import numpy as np
import random
import os
import subprocess

# Define the root path where the .npy files are stored
txt_root = './datasets/CREStereo_dataset'
image1_path = os.path.join(txt_root, "image1_list.npy")
image2_path = os.path.join(txt_root, "image2_list.npy")
disp_path = os.path.join(txt_root, "disp_list.npy")

# Load file paths from the .npy files
image1_list = np.load(image1_path)
image2_list = np.load(image2_path)
disp_list = np.load(disp_path)

# Set the number of random samples to select
num_samples = 10  # Change this number as needed
selected_indices = random.sample(range(len(image1_list)), num_samples)

# Collect paths of selected files
selected_files = []
for i in selected_indices:
    selected_files.extend([image1_list[i], image2_list[i], disp_list[i]])
    # print(selected_files[-1])

# Upload selected files to cloud storage
# Ensure 'rclone' remote storage is configured, e.g., 'my_remote'
remote_path = "alist:/xunlei_private/Vis/CREStereo"

for file_path in selected_files:
    # Extract the parent directory name and the file name
    parent_dir = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    # Create a new file name by concatenating parent directory name and file name
    new_file_name = f"{parent_dir}-{file_name}"
    print("copy {} to {}".format(file_path, f"{remote_path}/{new_file_name}"))
    
    # Upload to cloud with the new name
    subprocess.run(["rclone", "copyto", file_path, f"{remote_path}/{new_file_name}"])

