import os
import re
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

root='./datasets/NerfStereo'

# 假设 image1_list、image2_list、disp_list 是你的文件路径列表
left_list = sorted(glob(os.path.join(root, "*/*/baseline_*/left/*.jpg"), recursive=True))
image1_list = []
vad_list    = []
for path in left_list:
    match = re.search(r"(.*?/Q/)", path)
    prefix = match.group(1)  # prefix
    suffix = os.path.basename(path)  # file name
    image1_list.append( f"{prefix}center/{suffix}" )
    suffix = suffix.replace(".jpg", ".png")
    vad_list.append( f"{prefix}AO/{suffix}" )
image2_list = sorted(glob(os.path.join(root, "*/*/baseline_*/right/*.jpg"), recursive=True))
disp_list = sorted(glob(os.path.join(root, "*/*/baseline_*/disparity/*.png"), recursive=True))
print("原始文件数量:", len(image1_list), len(image2_list), len(disp_list), len(vad_list))
# print(image1_list[10], image2_list[10], disp_list[10], vad_list[10], sep="\r\n")


# 使用一个函数去重并保持顺序
def remove_duplicates(paths):
    seen = set()
    unique_paths = []
    for path in paths:
        basename = os.path.basename(path).replace(".jpg", "").replace(".png", "")
        if basename not in seen:
            unique_paths.append(path)
            seen.add(basename)
    return unique_paths

# 去重后的路径列表
unique_image1_list = remove_duplicates(image1_list)
unique_image2_list = remove_duplicates(image2_list)
unique_disp_list = remove_duplicates(disp_list)
unique_vad_list = remove_duplicates(vad_list)

# 创建场景名称的集合，确保每个列表中的数据匹配
image1_scenes = {os.path.basename(path).replace(".jpg", "") for path in unique_image1_list}
image2_scenes = {os.path.basename(path).replace(".jpg", "") for path in unique_image2_list}
disp_scenes   = {os.path.basename(path).replace(".png", "") for path in unique_disp_list}
vad_scenes    = {os.path.basename(path).replace(".png", "") for path in unique_vad_list}
# print(sorted(list(image1_scenes))[0], 
#       sorted(list(image2_scenes))[0],
#       sorted(list(disp_scenes))[0],
#       sorted(list(vad_scenes))[0])

# 找到三个列表都存在的场景名称
valid_scenes = image1_scenes & image2_scenes & disp_scenes & vad_scenes

# 过滤出有效的文件路径
valid_image1_list = [path for path in image1_list if os.path.basename(path).replace(".jpg", "") in valid_scenes]
valid_image2_list = [path for path in image2_list if os.path.basename(path).replace(".jpg", "") in valid_scenes]
valid_disp_list   = [path for path in disp_list if os.path.basename(path).replace(".png", "") in valid_scenes]
valid_vad_list    = [path for path in vad_list if os.path.basename(path).replace(".png", "") in valid_scenes]

# 更新原列表
image1_list, image2_list, disp_list, vad_list = valid_image1_list, valid_image2_list, valid_disp_list, valid_vad_list

# 打印有效文件数量
print("有效文件数量:", len(image1_list), len(image2_list), len(disp_list), len(vad_list))
# print(image1_list[10], image2_list[10], disp_list[10], vad_list[10], sep="\r\n")



# 分割数据，设置进程数量
num_processes = 50

# 使用 numpy.array_split 保证每个分块大小相对均衡
image1_chunks = np.array_split(image1_list, num_processes)
image2_chunks = np.array_split(image2_list, num_processes)
disp_chunks = np.array_split(disp_list, num_processes)
vad_chunks = np.array_split(vad_list, num_processes)

def check_validity(img1_chunk, img2_chunk, disp_chunk, vad_chunks):
    valid_img1, valid_img2, valid_disp, valid_vad = [], [], [], []

    # cnt = 0
    for img1_path, img2_path, disp_path, vad_path in zip(img1_chunk, img2_chunk, disp_chunk, vad_chunks):
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 64.0
            vad  = cv2.imread(vad_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            disp = np.array(disp).astype(np.float32)
            vad  = np.array(vad).astype(np.float32)
        except Exception as err:
            print(err)
            print(f"文件失效：{img1_path}-{img1 is None} {img2_path}-{img2 is None} {disp_path}-{disp is None} {vad_path}-{vad is None}")
            continue
        
        if img1 is not None and img2 is not None and disp is not None and vad is not None:
            # if cnt==0:
            #     print(img1.shape==img2.shape and img2.shape==disp.shape and disp.shape==vad.shape, 
            #           img1.shape, img2.shape, disp.shape, vad.shape)
            #     cnt += 1
            if img1.shape[:2]==img2.shape[:2] and img2.shape[:2]==disp.shape[:2] and disp.shape[:2]==vad.shape[:2]:
                valid_img1.append(img1_path)
                valid_img2.append(img2_path)
                valid_disp.append(disp_path)
                valid_vad.append(vad_path)

    return valid_img1, valid_img2, valid_disp, valid_vad

# 使用进程池并行检查文件有效性
with Pool(processes=num_processes) as pool:
    # 主进程中使用 tqdm 显示整体进度
    results = list(tqdm(pool.starmap(check_validity, zip(image1_chunks, image2_chunks, disp_chunks, vad_chunks)), total=num_processes))

# 合并结果
valid_image1_list = [img for result in results for img in result[0]]
valid_image2_list = [img for result in results for img in result[1]]
valid_disp_list   = [img for result in results for img in result[2]]
valid_vad_list    = [img for result in results for img in result[3]]

# 更新列表
image1_list, image2_list, disp_list, vad_list = valid_image1_list, valid_image2_list, valid_disp_list, valid_vad_list

print("最终有效文件数量:", len(image1_list), len(image2_list), len(disp_list), len(vad_list))



# 将列表转换为 numpy 数组
image1_array = np.array(image1_list)
image2_array = np.array(image2_list)
disp_array   = np.array(disp_list)
vad_list     = np.array(vad_list)

# 分别保存为 .npy 文件
np.save('image1_list.npy', image1_array)
np.save('image2_list.npy', image2_array)
np.save('disp_list.npy', disp_array)
np.save('vad_list.npy', vad_list)

print("数据已保存为 .npy 文件")



