import os
import numpy as np
from glob import glob
import argparse

def build_index(file_list):
    """
    Build an index from (dataset_name, frame_id) -> file path.
    Example: ("amr_v5-b2_chaos_2500_1", "0011") -> "amr_v5-b2_chaos_2500_1/dataset/data/left/rgb/0011.png"
    """
    index = {}
    for f in file_list:
        parts = f.split(os.sep)
        dataset_name = parts[0]  # e.g., "amr_v5-b2_chaos_2500_1"
        frame_id = os.path.splitext(parts[-1])[0]  # remove ".png"
        index[(dataset_name, frame_id)] = f
    return index

def main(dataset_root, sv_root):
    # Collect all candidate files
    left_rgb_files  = [os.path.relpath(f, dataset_root) for f in glob(os.path.join(dataset_root, "**/left/rgb/*.jpg"), recursive=True)]
    right_rgb_files = [os.path.relpath(f, dataset_root) for f in glob(os.path.join(dataset_root, "**/right/rgb/*.jpg"), recursive=True)]
    disp_files      = [os.path.relpath(f, dataset_root) for f in glob(os.path.join(dataset_root, "**/left/disparity/*.png"), recursive=True)]

    # Build dictionaries for fast lookup
    left_index = build_index(left_rgb_files)
    right_index = build_index(right_rgb_files)
    disp_index = build_index(disp_files)

    image1_list, image2_list, disp_list = [], [], []

    # Match files with the same (dataset_name, frame_id)
    for key in sorted(left_index.keys()):
        if key in right_index and key in disp_index:
            image1_list.append(left_index[key])   # left rgb
            image2_list.append(right_index[key])  # right rgb
            disp_list.append(disp_index[key])     # disparity

    # Save results as numpy arrays
    np.save(os.path.join(sv_root, "image1_list.npy"), np.array(image1_list))
    np.save(os.path.join(sv_root, "image2_list.npy"), np.array(image2_list))
    np.save(os.path.join(sv_root, "disp_list.npy"), np.array(disp_list))

    print(f"Done! {len(image1_list)} triplets found and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--save_root", type=str, required=True, help="Directory to save the output numpy files")
    args = parser.parse_args()
    main(args.dataset_root, args.save_root)
