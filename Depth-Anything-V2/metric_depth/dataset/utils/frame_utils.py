import os
import glob
import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())



def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid

def writeDispKITTI(filename, disp):
    disp = np.round(disp * 256).astype(np.uint16)
    # skimage.io.imsave(filename, disp)
    cv2.imwrite(filename, disp)

def load_mask_image(path):
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is not None:
        mask[mask>=128] = 255
        mask[mask<128] = 0
    return mask

def is_support_unreliable(ill_mask, sup_mask, threshold=0.5):
    """
    Determine whether the support region is unreliable due to excessive overlap with the illusion region.

    Parameters:
    ill_mask (np.ndarray): Binary mask (255 for valid, 0 for invalid) for the illusion region.
    sup_mask (np.ndarray): Binary mask (255 for valid, 0 for invalid) for the support region.
    threshold (float): Overlap ratio threshold; default is 0.5 (more overlap than non-overlap).

    Returns:
    bool: True if support region is unreliable, False otherwise.
    """
    # Compute the overlap area
    overlap_mask = (sup_mask == 255) & (ill_mask == 255)
    
    # Count the number of pixels in each region
    sup_area = np.count_nonzero(sup_mask == 255)
    overlap_area = np.count_nonzero(overlap_mask)

    # Determine if the support region is unreliable
    return overlap_area > sup_area * threshold

def loadMaskFooling3D(mask_dir, frame_name, valid):
    # Acquire mask paths
    mask_paths = glob.glob( os.path.join(mask_dir, f'{frame_name}*.jpg') )

    # Parse mask information
    mask_dict  = {}
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        info = os.path.splitext(mask_name)[0].split("-")
        obj_id = info[1]
        area_type = info[3]
        if obj_id not in mask_dict:
            mask_dict[obj_id] = {}
        mask_dict[obj_id][area_type] = mask_path
    
    # Load mask images
    mask_image_dict = {}
    area_types = ["illusion", "nonillusion"]
    obj_id_list = list(mask_dict.keys())
    for obj_id in obj_id_list:
        ill_mask_path = mask_dict[obj_id].get(area_types[0])
        sup_mask_path = mask_dict[obj_id].get(area_types[1])

        # Skip this illusion mask if mask does not exist
        if ill_mask_path is None or not os.path.exists(ill_mask_path):
            continue

        ill_mask = load_mask_image(ill_mask_path)
        # Skip this illusion mask if too few positive values in the mask
        if ill_mask is None or (ill_mask > 128).sum() < 100:
            # print(f"Too few positive values in the illusion mask: {ill_mask_path}")
            continue

        sup_mask = None
        if sup_mask_path is not None and os.path.exists(sup_mask_path):
            sup_mask = load_mask_image(sup_mask_path)
        # The sup_mask is invalid if too few positive values in the mask
        if sup_mask is not None and (sup_mask > 128).sum() < 100:
            # print(f"Too few positive values in the support mask: {sup_mask_path}")
            sup_mask = None
        # The sup_mask is unreliable when excessive overlap with the illusion region
        if sup_mask is not None and is_support_unreliable(ill_mask, sup_mask, threshold=0.5):
            # print(f"The sup_mask is unreliable: {sup_mask_path}")
            sup_mask = None

        mask_image_dict[obj_id] = {}
        mask_image_dict[obj_id][area_types[0]] = ill_mask
        mask_image_dict[obj_id][area_types[1]] = sup_mask
    
    # Merge all masks
    mask = np.zeros_like(valid, dtype=bool)
    for obj_id in obj_id_list:
        ill_mask = mask_image_dict.get(obj_id, {}).get(area_types[0])
        sup_mask = mask_image_dict.get(obj_id, {}).get(area_types[1])

        if ill_mask is not None:
            mask[ill_mask > 128] = True
        if sup_mask is not None:
            mask[sup_mask > 128] = True
    
    if mask.sum() < 100:
        mask = None
    
    return mask

def readDispFooling3D(filename, mask_path=None):
    if os.path.splitext(filename)[-1].lower() in [".jpg", ".png", ".jpeg"]:
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    elif os.path.splitext(filename)[-1].lower() == ".pfm":
        disp = readPFM(filename).astype(np.float32)
    
    try:
        valid = disp > 0.0
    except Exception as err:
        raise(Exception(err, "Something wrong with {}".format(filename), os.getcwd()))

    if mask_path is not None:
        mask = load_mask_image(mask_path)
    else:
        tmp_path = filename.replace("depth_rect", "sam_mask")
        mask_dir = os.path.dirname(tmp_path)
        frame_name = os.path.splitext(os.path.basename(tmp_path))[0]
        mask = loadMaskFooling3D(mask_dir, frame_name, valid)
    
    if mask is not None:
        valid = valid & mask

    # try:
    #     assert np.sum(valid) > 100, f"Invalid disp: {filename}, shape: {valid.shape}, valid: {np.sum(valid)}, mask: {np.sum(mask)}, disp: {np.sum(disp>0.0)}"
    # except Exception as err:
    #     cv2.imwrite("./disp_mask.jpg", (disp > 0.0).astype(np.uint8) * 255)
    #     cv2.imwrite("./sam_mask.jpg", mask.astype(np.uint8) * 255)
    #     raise Exception(err)

    return disp, valid

def writeDispFooling3D(filename, disp):
    disp = np.round(disp).astype(np.uint16)
    # skimage.io.imsave(filename, disp)
    cv2.imwrite(filename, disp)


def readDispCRES(filename):
    try:
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 32.0
        valid = disp > 0.0
        return disp, valid
    except Exception as err:
        raise(Exception(err, "Something wrong with {}".format(filename), os.getcwd()))

def writeDispCRES(filename, disp):
    disp = np.round(disp * 32).astype(np.uint16)
    # skimage.io.imsave(filename, disp)
    cv2.imwrite(filename, disp)

def readDispNerfS(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 64.0
    
    match = re.search(r"(.*?/Q/)", filename)
    if match:
        prefix = match.group(1)  # prefix
        suffix = os.path.basename(filename)  # file name
        # AO path, aka confidence
        ao_path = f"{prefix}AO/{suffix}"
        # print("AO图路径:", ao_path)
    else:
        raise Exception("corrupted path for NerfStereo: {}".format(filename))
    valid = cv2.imread(ao_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535
    return disp, valid

def writeDispNerfS(filename, disp):
    disp = np.round(disp * 64).astype(np.uint16)
    # skimage.io.imsave(filename, disp)
    cv2.imwrite(filename, disp)

def readDispBooster(file_name):
    disp = np.load(file_name, encoding='bytes', allow_pickle=True)
    # mask_00  = os.path.join(os.path.split(file_name)[0], 'mask_00.png')
    mask_cat_path = os.path.join(os.path.split(file_name)[0], 'mask_cat.png')
    mask_cat = cv2.imread(mask_cat_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    valid = mask_cat
    return disp, valid

def writeDispBooster(filename, disp):
    # disp = np.round(disp).astype(np.uint16)
    # # skimage.io.imsave(filename, disp)
    # filename = filename.replace(".npy", ".jpg")
    # cv2.imwrite(filename, disp)
    np.save(filename, disp)

# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2**6) + d_b / (2**14))[..., 0]
    mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid

# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open('/'.join(file_name.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid

# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid


def readDispMiddlebury(file_name):
    if basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    elif basename(file_name) == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid

def writeDispMiddlebury(file_name, disp):
    writePFM(file_name, disp)

def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

def write_gen(file_name, disp, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        raise Exception("no support for {} file".format(ext))
    elif ext == '.bin' or ext == '.raw':
        np.save(disp, file_name)
    elif ext == '.flo':
        writeFlow(file_name, disp)
    elif ext == '.pfm':
        writePFM(file_name, disp)
    else:
        raise Exception("no support for {} file".format(ext))