import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

root='./datasets/CREStereo_dataset'

# 假设 image1_list、image2_list、disp_list 是你的文件路径列表
image1_list = sorted(glob(os.path.join(root, "**/*_left.jpg"), recursive=True))
image2_list = sorted(glob(os.path.join(root, "**/*_right.jpg"), recursive=True))
disp_list = sorted(glob(os.path.join(root, "**/*_left.disp.png"), recursive=True))
print("原始文件数量:", len(image1_list))



# 创建场景名称的集合，确保每个列表中的数据匹配
image1_scenes = {os.path.basename(path).replace("_left.jpg", "") for path in image1_list}
image2_scenes = {os.path.basename(path).replace("_right.jpg", "") for path in image2_list}
disp_scenes = {os.path.basename(path).replace("_left.disp.png", "") for path in disp_list}

# 找到三个列表都存在的场景名称
valid_scenes = image1_scenes & image2_scenes & disp_scenes

# 过滤出有效的文件路径
valid_image1_list = [path for path in image1_list if os.path.basename(path).replace("_left.jpg", "") in valid_scenes]
valid_image2_list = [path for path in image2_list if os.path.basename(path).replace("_right.jpg", "") in valid_scenes]
valid_disp_list = [path for path in disp_list if os.path.basename(path).replace("_left.disp.png", "") in valid_scenes]

# 更新原列表
image1_list, image2_list, disp_list = valid_image1_list, valid_image2_list, valid_disp_list

# 打印有效文件数量
print("有效文件数量:", len(image1_list))



# 分割数据，设置进程数量
num_processes = 50

# 使用 numpy.array_split 保证每个分块大小相对均衡
image1_chunks = np.array_split(image1_list, num_processes)
image2_chunks = np.array_split(image2_list, num_processes)
disp_chunks = np.array_split(disp_list, num_processes)

def check_validity(img1_chunk, img2_chunk, disp_chunk):
    valid_img1, valid_img2, valid_disp = [], [], []

    for img1_path, img2_path, disp_path in zip(img1_chunk, img2_chunk, disp_chunk):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        disp = cv2.imread(disp_path)

        # 检查是否有任何图像读取失败
        if img1 is not None and img2 is not None and disp is not None:
            valid_img1.append(img1_path)
            valid_img2.append(img2_path)
            valid_disp.append(disp_path)
        else:
            print(f"文件失效：{img1_path if img1 is None else ''} {img2_path if img2 is None else ''} {disp_path if disp is None else ''}")

    return valid_img1, valid_img2, valid_disp

# 使用进程池并行检查文件有效性
with Pool(processes=num_processes) as pool:
    # 主进程中使用 tqdm 显示整体进度
    results = list(tqdm(pool.starmap(check_validity, zip(image1_chunks, image2_chunks, disp_chunks)), total=num_processes))

# 合并结果
valid_image1_list = [img for result in results for img in result[0]]
valid_image2_list = [img for result in results for img in result[1]]
valid_disp_list = [img for result in results for img in result[2]]

# 更新列表
image1_list, image2_list, disp_list = valid_image1_list, valid_image2_list, valid_disp_list

print("最终有效文件数量:", len(image1_list))



# 将列表转换为 numpy 数组
image1_array = np.array(image1_list)
image2_array = np.array(image2_list)
disp_array = np.array(disp_list)

# 分别保存为 .npy 文件
np.save('image1_list.npy', image1_array)
np.save('image2_list.npy', image2_array)
np.save('disp_list.npy', disp_array)

print("数据已保存为 .npy 文件")



