import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

def get_middlebury_images():
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), 'r') as f:
        lines = f.read().splitlines()
    return sorted([os.path.join(root, 'trainingQ', f'{name}/im0.png') for name in lines])

def get_eth3d_images():
    return sorted(glob('datasets/ETH3D/two_view_training/*/im0.png'))

def get_kitti_images():
    return sorted(glob('datasets/KITTI/training/image_2/*_10.png'))

def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0,1), keepdims=True)# + 1
    reference_mean = np.mean(reference_image_lab, axis=(0,1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev/reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l,a,b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb

class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob or min_scale > 1:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow


    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1], resize=None):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.resize = resize

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def pad_images(self, img1, img2, flow, valid):
        ch, cw = self.crop_size
        padded_data = []

        for data in [img1, img2, flow, valid]:
            h, w = data.shape[:2]
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            
            if pad_h > 0 or pad_w > 0:
                pad_width = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (data.ndim - 2)
                padded_data.append(np.pad(data, pad_width, mode='constant', constant_values=0))
            else:
                padded_data.append(data)

        return padded_data

    def get_crop_start_point(self, valid, expand_ratio=1/3):
        # Assume img1, img2, flow, and valid are already defined
        H, W = valid.shape[:2]
        crop_h, crop_w = self.crop_size

        # Get coordinates where valid == 1
        valid_y, valid_x = np.where(valid == 1)

        # Ensure there is a valid region to crop
        if len(valid_y) == 0 or len(valid_x) == 0:
            return 0, 0

        # Compute the bounding box of the valid region
        y_min, y_max = np.min(valid_y), np.max(valid_y)
        x_min, x_max = np.min(valid_x), np.max(valid_x)

        # Calculate the expansion amount
        expand_h = int(crop_h * expand_ratio)
        expand_w = int(crop_w * expand_ratio)

        # Adjust the crop range to allow partial expansion outside the valid region
        y_min = max(0, y_min - expand_h)  # Expand upward
        y_max = min(H - crop_h, max(0, y_max - crop_h + expand_h))  # Expand downward

        x_min = max(0, x_min - expand_w)  # Expand leftward
        x_max = min(W - crop_w, max(0, x_max - crop_w + expand_w))  # Expand rightward

        # Ensure valid range by swapping if needed
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if x_min > x_max:
            x_min, x_max = x_max, x_min
            
        try:
            # Ensure the selected crop position is within the adjusted range
            y0 = np.random.randint(y_min, y_max + 1)
            x0 = np.random.randint(x_min, x_max + 1)
        except Exception as err:
            raise Exception(err, f"y_min: {y_min}, y_max: {y_max}, x_min: {x_min}, x_max: {x_max}, crop_h: {crop_h}, crop_w: {crop_w}, H: {H}, W: {W}, expand_h: {expand_h}, expand_w: {expand_w}, np.min(valid_y): {np.min(valid_y)}, np.max(valid_y): {np.max(valid_y)}, np.min(valid_x): {np.min(valid_x)}, np.max(valid_x): {np.max(valid_x)}")
            # y_min: 0, y_max: -126, x_min: 524, x_max: 1041, crop_h: 320, crop_w: 736, H: 999, W: 1777, expand_h: 106, expand_w: 245
        
        y0 = np.clip(y0, 0, H - crop_h)
        x0 = np.clip(x0, 0, W - crop_w)

        return x0, y0

    # # def spatial_transform(self, img1, img2, flow, valid):
    # def spatial_transform(self, img1, img2, flow, valid, path=None):
    #     # randomly sample scale

    #     ht, wd = img1.shape[:2]
    #     min_scale = np.maximum(
    #         (self.crop_size[0] + 1) / float(ht), 
    #         (self.crop_size[1] + 1) / float(wd))

    #     scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
    #     scale_x = np.clip(scale, min_scale, None)
    #     scale_y = np.clip(scale, min_scale, None)

    #     if path is not None and "3D_PAINTING_DESIGNS_3D_WALL_PAINTING_SIMPLE_3D_WALL_DECORATION_PAINTING/frame_0310" in path:
    #         # print("-"*10, "SparseFlowAugmentor: ", path, img1.shape, valid.shape, x0, y0)
    #         cv2.imwrite("./AUG1_image1.jpg", img1.astype(np.uint8))
    #         cv2.imwrite("./AUG1_valid.jpg", (valid >= 0.5).astype(np.uint8)*255)
    #         cv2.imwrite("./AUG1_flow.jpg", (np.abs(flow[...,0]) < 700).astype(np.uint8) * 255)

    #     if np.random.rand() < self.spatial_aug_prob:
    #         # rescale the images
    #         img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    #         img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    #         flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

    #     if path is not None and "3D_PAINTING_DESIGNS_3D_WALL_PAINTING_SIMPLE_3D_WALL_DECORATION_PAINTING/frame_0310" in path:
    #         # print("-"*10, "SparseFlowAugmentor: ", path, img1.shape, valid.shape, x0, y0)
    #         cv2.imwrite("./AUG2_image1.jpg", img1.astype(np.uint8))
    #         cv2.imwrite("./AUG2_valid.jpg", (valid >= 0.5).astype(np.uint8)*255)
    #         cv2.imwrite("./AUG2_flow.jpg", (np.abs(flow[...,0]) < 700).astype(np.uint8) * 255)

    #     if self.do_flip:
    #         if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
    #             img1 = img1[:, ::-1]
    #             img2 = img2[:, ::-1]
    #             flow = flow[:, ::-1] * [-1.0, 1.0]

    #         if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
    #             tmp = img1[:, ::-1]
    #             img1 = img2[:, ::-1]
    #             img2 = tmp

    #         if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
    #             img1 = img1[::-1, :]
    #             img2 = img2[::-1, :]
    #             flow = flow[::-1, :] * [1.0, -1.0]

    #     img1, img2, flow, valid = self.pad_images(img1, img2, flow, valid)
    #     # # img1_raw_shape = img1.shape
    #     # # valid_raw_shape = valid.shape

    #     # margin_y = 20
    #     # margin_x = 50
    #     # y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
    #     # x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

    #     # y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
    #     # x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

    #     x0, y0 = self.get_crop_start_point(valid, expand_ratio=1/3)

    #     img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
    #     img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
    #     flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
    #     valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

    #     # print("-"*10, "SparseFlowAugmentor: ", self.crop_size, [x0,y0], img1.shape, img1_raw_shape, valid.shape, valid_raw_shape)
    #     return img1, img2, flow, valid
    
    def spatial_transform(self, img1, img2, flow, valid, path=None):
        # randomly sample scale
        img1 = cv2.resize(img1, self.resize, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, self.resize, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, self.resize, interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), self.resize, interpolation=cv2.INTER_NEAREST)
        return img1, img2, flow, valid


    # def __call__(self, img1, img2, flow, valid):
    def __call__(self, img1, img2, flow, valid, path=None):
        if path is not None and "3D_PAINTING_DESIGNS_3D_WALL_PAINTING_SIMPLE_3D_WALL_DECORATION_PAINTING/frame_0310" in path:
            # from PIL import Image
            # Image.fromarray(img1.astype(np.uint8)).save("AUG_image1.png")
            # Image.fromarray(((valid >= 0.5)*255).astype(np.uint8)).save("AUG_valid.png")
            # Image.fromarray(((np.abs(flow[:1]) < 700)*255).astype(np.uint8)).save("AUG_flow.png")
            # import cv2
            cv2.imwrite("./CALL_img1.jpg", img1.astype(np.uint8))
            cv2.imwrite("./CALL_mask.jpg", (valid>=0.5).astype(np.uint8) * 255)
            cv2.imwrite("./CALL_flow.jpg", (np.abs(flow[..., 0]) < 700).astype(np.uint8) * 255)
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid, path)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
