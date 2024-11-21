import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

root = './datasets/CREStereo_dataset'

# Assume image1_list, image2_list, disp_list are your file path lists
image1_list = sorted(glob(os.path.join(root, "**/*_left.jpg"), recursive=True))
image2_list = sorted(glob(os.path.join(root, "**/*_right.jpg"), recursive=True))
disp_list = sorted(glob(os.path.join(root, "**/*_left.disp.png"), recursive=True))
print("Original number of files:", len(image1_list))

# Create a set of scene names to ensure data in each list matches
image1_scenes = {os.path.basename(path).replace("_left.jpg", "") for path in image1_list}
image2_scenes = {os.path.basename(path).replace("_right.jpg", "") for path in image2_list}
disp_scenes = {os.path.basename(path).replace("_left.disp.png", "") for path in disp_list}

# Find scene names that exist in all three lists
valid_scenes = image1_scenes & image2_scenes & disp_scenes

# Filter out valid file paths
valid_image1_list = [path for path in image1_list if os.path.basename(path).replace("_left.jpg", "") in valid_scenes]
valid_image2_list = [path for path in image2_list if os.path.basename(path).replace("_right.jpg", "") in valid_scenes]
valid_disp_list = [path for path in disp_list if os.path.basename(path).replace("_left.disp.png", "") in valid_scenes]

# Update the original lists
image1_list, image2_list, disp_list = valid_image1_list, valid_image2_list, valid_disp_list

# Print the number of valid files
print("Number of valid files:", len(image1_list))

# Split the data and set the number of processes
num_processes = 50

# Use numpy.array_split to ensure relatively balanced chunk sizes
image1_chunks = np.array_split(image1_list, num_processes)
image2_chunks = np.array_split(image2_list, num_processes)
disp_chunks = np.array_split(disp_list, num_processes)

def check_validity(img1_chunk, img2_chunk, disp_chunk):
    valid_img1, valid_img2, valid_disp = [], [], []

    for img1_path, img2_path, disp_path in zip(img1_chunk, img2_chunk, disp_chunk):
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 64.0

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            disp = np.array(disp).astype(np.float32)
        except Exception as err:
            print(err)
            print(f"Invalid file: {img1_path if img1 is None else ''} {img2_path if img2 is None else ''} {disp_path if disp is None else ''}")
            continue
        
        if img1 is not None and img2 is not None and disp is not None:
            valid_img1.append(img1_path)
            valid_img2.append(img2_path)
            valid_disp.append(disp_path)
        
    return valid_img1, valid_img2, valid_disp

# Use a process pool to check file validity in parallel
with Pool(processes=num_processes) as pool:
    # Show overall progress in the main process with tqdm
    results = list(tqdm(pool.starmap(check_validity, zip(image1_chunks, image2_chunks, disp_chunks)), total=num_processes))

# Combine results
valid_image1_list = [img for result in results for img in result[0]]
valid_image2_list = [img for result in results for img in result[1]]
valid_disp_list = [img for result in results for img in result[2]]

# Update the lists
image1_list, image2_list, disp_list = valid_image1_list, valid_image2_list, valid_disp_list

print("Final number of valid files:", len(image1_list))

# Convert the lists to numpy arrays
image1_array = np.array(image1_list)
image2_array = np.array(image2_list)
disp_array = np.array(disp_list)

# Save them as .npy files
np.save('image1_list.npy', image1_array)
np.save('image2_list.npy', image2_array)
np.save('disp_list.npy', disp_array)

print("Data has been saved as .npy files")
