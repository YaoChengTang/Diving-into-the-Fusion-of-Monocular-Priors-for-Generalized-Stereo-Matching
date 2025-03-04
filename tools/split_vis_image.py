import cv2
import numpy as np
import os
from multiprocessing import Pool



def pad_to_match_height(image, target_height, color=(255, 255, 255)):
    h, w, c = image.shape
    if h >= target_height:
        return image  # No padding needed
    top_pad = 0
    bottom_pad = target_height - h - top_pad
    return cv2.copyMakeBorder(image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=color)

def compress(img, resize_factor, path):
    if image_path.find("kitti")!=-1:
        return img
    if image_path.find("eth3d")!=-1:
        return img
    if image_path.find("middlebury")!=-1:
        return np.hstack([img[:, 100:650, :], 
                          img[:, 850:1400, :], 
                          img[:, 1600:2150, :]])
    elif image_path.find("booster")!=-1:
        return np.hstack([img[:, int(590 * resize_factor):int(2550 * resize_factor), :], 
                          img[:, int(3500 * resize_factor):int(5550 * resize_factor), :], 
                          img[:, int(6450 * resize_factor):int(8450 * resize_factor), :]])
    else:
        raise Exception("Not supported: ", path)

def split_image_with_custom_grouping(image_path, output_dir, num_rows, resize_factor=None, compression_params=None, gap=10, destination_path=None):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read the image from {image_path}")
    
    if resize_factor is not None:
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    # cv2.imwrite(os.path.join(output_dir, "resized_image.png"), img)

    # Get the height and width of the image
    height, width, _ = img.shape

    # Determine the height of each row
    row_height = height // num_rows

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Combine the first five rows
    first_five_height = 5 * row_height
    first_five_rows = img[:first_five_height, :, :]
    if image_path.find("kitti")!=-1:
        group = []
        for i in range(0, 5):
            upper = i * row_height + 50
            lower = (i + 1) * row_height - 40
            group.append(img[upper:lower, :, :])
        first_five_rows = np.vstack(group)
    compressed_first_five_rows = compress(first_five_rows, resize_factor, image_path)
    # cv2.imwrite(os.path.join(output_dir, "first_five_rows.jpg"), compressed_first_five_rows, compression_params)

    # Initialize groups
    group_0 = []  # Rows divisible by 3
    group_1 = []  # Rows with remainder 1 when divided by 3
    group_2 = []  # Rows with remainder 2 when divided by 3
    group_3 = []  # Additional rows with remainder 2

    upper_shift, lower_shit = 0, 0
    if image_path.find("kitti")!=-1:
        upper_shift = 50
        lower_shit = -40

    # Process the remaining rows
    for i in range(5, num_rows):
        if i < 29:
            upper = i * row_height + upper_shift
            lower = (i + 1) * row_height + lower_shit if i < num_rows - 1 else height
        else:
            upper = 29 * row_height + (i - 29) * (row_height - 1) + upper_shift
            lower = 29 * row_height + (i - 28) * (row_height - 1) + lower_shit if i < num_rows - 1 else height

        row_img = img[upper:lower, :, :]

        compressed_row_img = compress(row_img, resize_factor, image_path)
        
        if i < num_rows - 2:
            # Group rows based on modulo 3
            if (i - 5) % 3 == 0:
                group_0.append(compressed_row_img)
            elif (i - 5) % 3 == 1:
                group_1.append(compressed_row_img)
            else:
                group_2.append(compressed_row_img)
        else:
            group_3.append(compressed_row_img)

    # Concatenate and save each group
    if group_0:
        group_0_img = np.vstack(group_0)  # Concatenate vertically
        # cv2.imwrite(os.path.join(output_dir, "group_0.jpg"), group_0_img, compression_params)

    if group_1:
        group_1_img = np.vstack(group_1)
        # cv2.imwrite(os.path.join(output_dir, "group_1.jpg"), group_1_img, compression_params)

    if group_2:
        group_2_img = np.vstack(group_2)
        # cv2.imwrite(os.path.join(output_dir, "group_2.jpg"), group_2_img, compression_params)
    
    if group_3:
        group_3_img = np.vstack(group_3)
        # cv2.imwrite(os.path.join(output_dir, "group_3.jpg"), group_3_img, compression_params)
    

    if destination_path is not None:
        group_images = []
        group_images.append(np.vstack(group_1))
        group_images.append(np.vstack(group_0))
        group_images.append(np.vstack(group_2))

        # Create the gap as a black (zeros) image
        gap_img = np.ones((gap, group_3_img.shape[1], 3), dtype=np.uint8) * 255
        left_img = np.vstack([compressed_first_five_rows, gap_img, group_3_img])

        # Find the maximum height among all images
        max_height = max(left_img.shape[0], *(img.shape[0] for img in group_images))
        
        # Pad the first image to match the maximum height
        padded_image1 = pad_to_match_height(left_img, max_height)
        padded_images = [pad_to_match_height(img, max_height) for img in group_images]

        gap_img = np.ones((max_height, gap, 3), dtype=np.uint8) * 255
        concatenated_img = padded_image1
        for img in padded_images:
            concatenated_img = np.hstack([concatenated_img, gap_img, img])
        
        # Save the result
        cv2.imwrite(destination_path.replace("png","jpg"), concatenated_img, compression_params)


def process_file(args):
    source_path, output_dir, num_rows, resize_factor, compression_params, gap, destination_path = args
    split_image_with_custom_grouping(source_path, output_dir, num_rows, resize_factor, compression_params, gap, destination_path)
    print(f"Processed {source_path} to {destination_path}")


def process(input_dir, output_dir, num_rows, resize_factor=None, compression_params=None, gap = 10):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    tasks = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is an image based on extension
            if file.lower().endswith(('.png',)):
                # Determine the relative path of the current directory
                relative_path = os.path.relpath(root, input_dir)
                
                # Create the corresponding output subdirectory
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Copy the file to the output directory
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_subdir, file.replace(".png", ".jpg"))

                tasks.append((source_path, output_dir, num_rows, resize_factor, compression_params, gap, destination_path))
                
                # split_image_with_custom_grouping(source_path, output_dir, num_rows, resize_factor, compression_params, gap, destination_path)
                # print(f"process {source_path} to {destination_path}")
    
    with Pool(processes=(os.cpu_count())//2) as pool:  # Use as many processes as available CPU cores
        pool.map(process_file, tasks)


image_path = "/data5/yao/runs/vis/inter_results/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/SUPP/analysis/booster/train-balanced-Case-disp_00.png"
# image_path = "/data5/yao/runs/vis/inter_results/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/SUPP/analysis/eth3d/two_view_training-delivery_area_1l-disp0GT.png"
# image_path = "/data5/yao/runs/vis/inter_results/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/SUPP/analysis/kitti/training-disp_occ_0-000000_10.png"
# image_path = "/data5/yao/runs/vis/inter_results/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/SUPP/analysis/middlebury_h/MiddEval3-trainingH-Adirondack-disp0GT.png"
output_dir = "./output"
num_rows = 43
resize_factor = 0.25
compression_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
destination_path = "./output/res.png"

# split_image_with_custom_grouping(image_path, output_dir, num_rows, resize_factor, compression_params, 10, destination_path)

input_dir = "/data5/yao/runs/vis/inter_results/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/SUPP/analysis/"
output_dir = "./output"
process(input_dir, output_dir, num_rows, resize_factor, compression_params, gap=10)

