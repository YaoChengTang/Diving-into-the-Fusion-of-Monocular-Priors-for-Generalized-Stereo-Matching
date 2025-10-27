# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import sys
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import argparse
from torch.utils.data.distributed import DistributedSampler

# from core.utils import plane
try:
    from core.utils import frame_utils
    from core.utils.ddp import get_loader
    from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
except:
    sys.path.insert(0, Path.cwd().as_posix())
    print("sys.path: ", sys.path)
    from core.utils import frame_utils
    from core.utils.ddp import get_loader
    from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


DATASET_ROOT = os.getenv('DATASET_ROOT')
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)



class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, args=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        # if args is not None:
        #     # self.plane = args.plane_datset
        #     self.slant = args.slant 
        #     self.slant_norm = args.slant_norm
        # else:
        #     # self.plane = False
        #     self.slant = None 
        #     self.slant_norm = False

        self.is_test = args.is_test if hasattr(args, "is_test") and args.is_test else False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = {}

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return self.image_list[index] + [self.disparity_list[index]], \
                   img1, img2, torch.zeros_like(torch.zeros_like(img1))[:1], torch.ones_like(torch.zeros_like(img1))[:1]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        try:
            index = index % len(self.image_list)
            intrinsic = self.extra_info["intrinsics"][index] if "intrinsics" in self.extra_info else None
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
        
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            disp = np.array(disp).astype(np.float32)
            flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        except Exception as err:
            raise Exception(err, "{}, {}, {}".format(self.image_list[index][0], 
                                                     self.image_list[index][1], 
                                                     self.disparity_list[index] ))

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid, intrinsic = self.augmentor(img1, img2, flow, valid, intrinsic)
            else:
                img1, img2, flow, intrinsic = self.augmentor(img1, img2, flow, intrinsic)

        try:
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            intrinsic = torch.from_numpy(np.array(intrinsic)).float() \
                if intrinsic is not None else torch.from_numpy(np.array([1, 1, 1, 1])).float()
        except Exception as err:
            raise Exception(err, "{}, {}, {}".format(self.image_list[index][0], 
                                                     self.image_list[index][1], 
                                                     self.disparity_list[index]),
                            "{}, {}, {}".format(img1.shape, img2.shape, flow.shape), )

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index]], \
               img1, img2, flow, valid.float(), intrinsic


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        if isinstance(copy_of_self.extra_info, list):
            copy_of_self.extra_info = v * copy_of_self.extra_info
        else:
            copy_of_self.extra_info = {key: val*v for key, val in copy_of_self.extra_info.items()}
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='', dstype='frames_cleanpass', 
                 things_test=False, caching=False, args=None, eval=False):
        super(SceneFlowDatasets, self).__init__(aug_params, args=args)
        self.eval = args.eval if args is not None else eval
        self.root = root if len(root)>0 else DATASET_ROOT
        self.dstype = dstype
        self.caching = caching
        self.extra_info["intrinsics"] = []
        assert os.path.exists(self.root), "check the existence: {}".format(self.root)

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        cache_file = osp.join(self.root, 'flying3d'+"-"+self.dstype+"-"+split+".npz")
        if self.caching and os.path.exists(cache_file):
            cache = np.load(cache_file)
            root = cache["root"]
            left_images = cache["left_images"]
            right_images = cache["right_images"]
            disparity_images = cache["disparity_images"]
        else :
            root = osp.join(self.root, 'flying3d')
            left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
            right_images = [ im.replace('left', 'right') for im in left_images ]
            disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
            if self.caching :
                np.savez(cache_file, 
                        root=root,
                        left_images=left_images, 
                        right_images=right_images, 
                        disparity_images=disparity_images)

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        if not self.eval:
            val_idxs = set(np.random.permutation(len(left_images))[:400])
        else:
            val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                self.extra_info["intrinsics"] += [ [1050, 1050, 479.5, 269.5] ]
        
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            self.extra_info["intrinsics"] += [ [1050, 1050, 479.5, 269.5] ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            if img1.find("15mm_focallength") != -1:
                self.extra_info["intrinsics"] += [ [450, 450, 479.5, 269.5] ]
            elif img1.find("35mm_focallength") != -1:
                self.extra_info["intrinsics"] += [ [1050, 1050, 479.5, 269.5] ]
            else:
                raise Exception(f"Unknown intrinsics: {im1}")
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training', args=None):
        super(ETH3D, self).__init__(aug_params, sparse=True, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo', args=None):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo, args=args)
        root = root if len(root)>0 else DATASET_ROOT

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings', args=None):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training', args=None):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class KITTI2012(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI2012', image_set='training', args=None):
        super(KITTI2012, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F', image_set='training', args=None):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, f"MiddEval3/{image_set}{split}/*"))))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/disp0GT.pfm') for name in lines])
            if image_set=="training":
                assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, root, image_set, split]
            else:
                assert len(image1_list) == len(image2_list) > 0, [image1_list, root, image_set, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]


class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/booster/train/balanced', image_set='train', args=None):
        super(Booster, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster)
        assert os.path.exists(root), print(root)
        # image1_list = sorted(glob(os.path.join(root, image_set, "**/camera_00/im*.png"), recursive=True))
        image2_list = sorted(glob(os.path.join(root, image_set, "**/camera_02/im*.png"), recursive=True))
        image1_list = [img.replace("camera_02", "camera_00") for img in image2_list]

        disp_list = [os.path.join(os.path.split(x)[0].replace("camera_00", ""), 'disp_00.npy') for x in image1_list]
        mask_list = [os.path.join(os.path.split(x)[0].replace("camera_00", ""), 'mask_cat.png') for x in image1_list]
        right_disp_list = [os.path.join(os.path.split(x)[0].replace("camera_00", ""), 'disp_02.npy') for x in image1_list]
        
        for img1, img2, disp, disp_r, mask in zip(image1_list, image2_list, disp_list, right_disp_list,mask_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
            # self.trans_mask += [mask]


class NerfStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/NerfStereo', image_set='training', args=None, txt_root=None):
        super(NerfStereoDataset, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispNerfS, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)
        
        if txt_root is None: 
            left_list = sorted(glob(os.path.join(root, "*/*/baseline_*/left/*.jpg"), recursive=True))
            image1_list = []
            for path in left_list:
                match = re.search(r"(.*?/Q/)", path)
                prefix = match.group(1)  # prefix
                suffix = os.path.basename(path)  # file name
                path_new = f"{prefix}center/{suffix}"
                image1_list.append( path_new )
            image2_list = sorted(glob(os.path.join(root, "*/*/baseline_*/right/*.jpg"), recursive=True))
            disp_list = sorted(glob(os.path.join(root, "*/*/baseline_*/disparity/*.png"), recursive=True))
            # dispr_list = sorted(glob(os.path.join(root, "**/*_right.disp.png"), recursive=True))
        else:
            image1_list = np.load( os.path.join(txt_root, 'image1_list.npy') )
            image2_list = np.load( os.path.join(txt_root, 'image2_list.npy') )
            disp_list = np.load( os.path.join(txt_root, 'disp_list.npy') )

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/CREStereo_dataset', image_set='training', args=None, txt_root=None):
        super(CREStereoDataset, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispCRES, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)

        if txt_root is None: 
            image1_list = sorted(glob(os.path.join(root, "**/*_left.jpg"), recursive=True))
            image2_list = sorted(glob(os.path.join(root, "**/*_right.jpg"), recursive=True))
            disp_list = sorted(glob(os.path.join(root, "**/*_left.disp.png"), recursive=True))
        else:
            image1_list = np.load( os.path.join(txt_root, 'image1_list.npy') )
            image2_list = np.load( os.path.join(txt_root, 'image2_list.npy') )
            disp_list = np.load( os.path.join(txt_root, 'disp_list.npy') )
        # dispr_list = sorted(glob(os.path.join(root, "**/*_right.disp.png"), recursive=True))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]



class FSDDataset(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FSD', image_set='training', args=None, txt_root=None, eval=False):
        super(FSDDataset, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispFSD, args=args)
        root = root if len(root)>0 else DATASET_ROOT
        assert os.path.exists(root), "check the existence: {}".format(root)

        if txt_root is None:
            image1_list = sorted(glob(os.path.join(root, "**/left/rgb/*.jpg"), recursive=True))
            image2_list = [img.replace("left/rgb", "right/rgb") for img in image1_list]
            disp_list   = [img.replace("left/rgb", "left/disparity").replace(".jpg", ".png") for img in image1_list]

        else:
            # 65000
            image1_list = [os.path.join(root, p) for p in np.load(os.path.join(txt_root, 'image1_list.npy'))]
            image2_list = [os.path.join(root, p) for p in np.load(os.path.join(txt_root, 'image2_list.npy'))]
            disp_list   = [os.path.join(root, p) for p in np.load(os.path.join(txt_root, 'disp_list.npy'))]

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        
        if eval:
            logging.info(f"Eval FSD, only use 2000 samples for quick validation")
            self.image_list = self.image_list[:2000]
            self.disparity_list = self.disparity_list[:2000]



class InfStereoDataset(StereoDataset):
    """Dataset class for InfinigenStereo (flying / indoor / nature)
    Each scene has the following structure:
        frames/
            ├── Image/
            │   ├── camera_0/Image_..._0.png  ← left view
            │   └── camera_1/Image_..._1.png  ← right view
            ├── disparity/
            │   └── camera_0/Image_..._0.npy  ← disparity map
            └── camview/
                └── camera_0/camview_..._0.npz ← camera intrinsics & extrinsics
    """

    def __init__(self, aug_params=None, root='datasets/InfinigenStereo/release_full',
                 image_set='training', args=None, eval=False, caching=True):
        super(InfStereoDataset, self).__init__(
            aug_params, sparse=False, reader=frame_utils.readDispInfStereo, args=args
        )

        self.root = root
        self.eval = eval
        self.caching = caching

        assert os.path.exists(self.root), f"[InfStereoDataset] Root path not found: {self.root}"

        cache_file = os.path.join(self.root, "InfStereo_cache.npz")

        # Load cache if available
        if self.caching and os.path.exists(cache_file):
            logging.info(f"[InfStereoDataset] Loading cache from {cache_file}")
            cache = np.load(cache_file, allow_pickle=True)
            image1_list = list(cache["image1_list"])
            image2_list = list(cache["image2_list"])
            disp_list = list(cache["disp_list"])
            cam_list = list(cache["cam_list"])
        else:
            logging.info(f"[InfStereoDataset] Scanning dataset under {self.root}")
            image1_list, image2_list, disp_list, cam_list = self._scan_dataset()

            if self.caching:
                np.savez(cache_file,
                         image1_list=image1_list,
                         image2_list=image2_list,
                         disp_list=disp_list,
                         cam_list=cam_list)
                logging.info(f"[InfStereoDataset] Cached {len(image1_list)} samples to {cache_file}")

        # Register all samples
        self.extra_info["cam"] = []
        for img1, img2, disp, cam_param in zip(image1_list, image2_list, disp_list, cam_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
            self.extra_info["cam"] += [cam_param["K"][0,0], cam_param["K"][1,1], 
                                       cam_param["K"][0,2], cam_param["K"][1,2], ]

        if eval:
            logging.info("[InfStereoDataset] Eval mode: using only first 2000 samples")
            self.image_list = self.image_list[:2000]
            self.disparity_list = self.disparity_list[:2000]
            self.extra_info["cam"] = self.extra_info["cam"][:2000]

        logging.info(f"[InfStereoDataset] Loaded {len(self.image_list)} stereo pairs from {self.root}")


    def _right_name_from_left(self, left_png_name: str) -> str:
        """Convert left image filename to right image filename
        Example: 'Image_0_0_0048_0.png' → 'Image_0_0_0048_1.png'
        """
        m = re.match(r"(.*)_(\d+)\.png$", left_png_name)
        if not m:
            return left_png_name  # fallback: unchanged
        prefix, view_id = m.groups()
        return f"{prefix}_1.png"

    def _disp_name_from_left(self, left_png_name: str) -> str:
        """Convert left image name to disparity file name
        Example: 'Image_0_0_0048_0.png' → 'Image_0_0_0048_0.npy'
        """
        return left_png_name.replace("Image_", "disparity_").replace(".png", ".npy")

    def _cam_name_from_left(self, left_png_name: str) -> str:
        """Convert left image name to camera parameter file name
        Example: 'Image_0_0_0048_0.png' → 'camview_0_0_0048_0.npz'
        """
        stem = left_png_name[:-4]  # remove .png
        if stem.startswith("Image"):
            stem = "camview" + stem[len("Image"):]
        return stem + ".npz"

    def _read_cam_npz(self, path: str) -> dict:
        """Read a camera parameter npz file and return {K, T, HW} dict"""
        try:
            data = np.load(path)
            return dict(
                K=data.get('K', None),
                T=data.get('T', None),
                HW=data.get('HW', None)
            )
        except Exception as e:
            logging.warning(f"[InfStereoDataset] Failed to read {path}: {e}")
            return dict(K=None, T=None, HW=None)

    def _scan_dataset(self):
        """Fast recursive scan with parallel I/O and scandir"""
        image1_list, image2_list, disp_list, cam_list = [], [], [], []
        categories = ["flying", "indoor", "nature"]
        show_progress = (LOCAL_RANK == 0)

        def process_scene(cat_dir, scene):
            frames_dir = cat_dir / scene / "frames"
            if not frames_dir.is_dir():
                return [], [], [], []

            left_dir  = frames_dir / "Image" / "camera_0"
            right_dir = frames_dir / "Image" / "camera_1"
            disp_dir  = frames_dir / "disparity" / "camera_0"
            cam_dir   = frames_dir / "camview" / "camera_0"

            if not (left_dir.is_dir() and right_dir.is_dir() and disp_dir.is_dir() and cam_dir.is_dir()):
                return [], [], [], []

            img1, img2, disp, cam = [], [], [], []
            for left_path in sorted(left_dir.glob("*.png")):
                left_name = left_path.name
                right_path = right_dir / self._right_name_from_left(left_name)
                disp_path = disp_dir / self._disp_name_from_left(left_name)
                cam_path = cam_dir / self._cam_name_from_left(left_name)
                if not (right_path.exists() and disp_path.exists() and cam_path.exists()):
                    missing = []
                    if not right_path.exists():
                        missing.append("right")
                    if not disp_path.exists():
                        missing.append("disp")
                    if not cam_path.exists():
                        missing.append("cam")
                    tqdm.write(f"[WARN] Missing {', '.join(missing)} for {left_path}")
                    continue
                img1.append(str(left_path))
                img2.append(str(right_path))
                disp.append(str(disp_path))
                cam.append(self._read_cam_npz(str(cam_path)))
            return img1, img2, disp, cam

        for cat in categories:
            cat_dir = Path(self.root) / cat
            if not cat_dir.is_dir():
                continue

            scenes = [p.name for p in cat_dir.iterdir() if p.is_dir()]
            iterator = tqdm(scenes, desc=f"Scanning {cat}", ncols=100) if show_progress else scenes

            with ThreadPoolExecutor(max_workers=8) as ex:
                for img1, img2, disp, cam in ex.map(lambda s: process_scene(cat_dir, s), iterator):
                    image1_list += img1
                    image2_list += img2
                    disp_list += disp
                    cam_list += cam

        return image1_list, image2_list, disp_list, cam_list




class Trans(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/Trans', things_test=False, args=None):
        super(Trans, self).__init__(aug_params)
        self.root = root if len(root)>0 else DATASET_ROOT
        self.args = args
        self.extra_info["intrinsics"] = []
        
        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")

    def _add_things(self, split='TRAIN'):
        original_length = len(self.disparity_list)

        left_images = sorted(glob(osp.join(self.root, split, '*/*/left/img/*.jpg')) )
        assert len(left_images)>0, f"Loaded 0 images from {self.root}"

        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace('img', 'disparity').replace('.jpg', '.pfm') for im in left_images ]
        disparity_images_noTran = [im.replace('img', 'disparity_without_trans').replace('.jpg', '.pfm') for im in left_images ]

        for idx, (img1, img2, disp, disp_noTran) in enumerate(zip(left_images, right_images, disparity_images, disparity_images_noTran)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            # self.multi_label.append([disp, disp_noTran])
            self.extra_info["intrinsics"] += [ [933.3333333333334, 787.5, 480.0, 270.0] ]
        logging.info("-"*10 + f"Added {len(self.disparity_list) - original_length} from Trans")

class Fooling3DDataset(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Fooling3D', image_set='training', args=None):
        super(Fooling3DDataset, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispFooling3D)
        assert os.path.exists(root)
        self.root = root
        self.image_set = image_set
        self.video_frames_info = {}
        
        self._add_mono()
        self._build_video_frames_info()

    def _add_mono(self):
        origin_length = len(self.disparity_list)
        print(f"using {self.image_set} in fooling3D")
        
        if self.image_set=="training":
            df = pd.read_csv(os.path.join(self.root, 'meta_data/scale_factors.csv'), header=None)

            # df.columns = ['path', 'scale']
            # video_name = "Service_Cars_1_deleted_scene_3d_remake_Servio_Comunitrio"
            # df = df[df['path'].str.contains(video_name, case=False, na=False)]

            self.scale_factor = dict(zip(
                df.iloc[:, 0].apply(lambda x: x.replace('/data2', './datasets')),
                df.iloc[:, 1]
            ))
            # right_images = sorted(glob(os.path.join(self.root, 'video_frame_sequence_right/*/*/*.png')))
            right_images = df.iloc[:, 0].apply(lambda x: x.replace('/data2', './datasets')).tolist()
            disp_list =  [ im.replace('video_frame_sequence_right', 'depth_rect') for im in right_images ]
            left_images = [ im.replace('video_frame_sequence_right', 'video_frame_sequence') for im in right_images ]

            assert len(left_images) == len(right_images) == len(disp_list) > 0, [len(left_images), len(right_images), len(disp_list)]
            for img1, img2, disp in zip(left_images, right_images, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        
        elif self.image_set=="testing":
            with open(os.path.join(self.root, 'meta_data/testing_enter.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            self.extra_info["mask"] = []
            for key, frame_dict in data.items():
                left_image_path  = os.path.join(self.root, "real_data/testing", frame_dict["left"])
                right_image_path = os.path.join(self.root, "real_data/testing", frame_dict["right"])
                disp_image_path  = os.path.join(self.root, "real_data/testing", frame_dict["disp"])
                mask_image_path  = os.path.join(self.root, "real_data/testing", frame_dict["mask"])

                self.image_list += [ [left_image_path, right_image_path] ]
                self.disparity_list += [ disp_image_path ]
                self.extra_info["mask"] += [ mask_image_path ]

            assert len(self.image_list) == len(self.disparity_list) == len(self.extra_info["mask"]) > 0, \
                   [len(self.image_list), len(self.disparity_list), len(self.extra_info["mask"])]

        else:
            raise Exception(f"{self.image_set} is not in ['training', 'testing']")
        
        logging.info(f"Added {len(self.disparity_list) - origin_length} from Fooling3D Mono")
    
    def _build_video_frames_info(self):
        for idx, img_path in enumerate(self.disparity_list):
            parts = img_path.split('/')
            video_name = parts[-2]
            frame_name = parts[-1]

            if video_name not in self.video_frames_info:
                self.video_frames_info[video_name] = []

            self.video_frames_info[video_name].append(idx)
        self.video_frames_info = list(self.video_frames_info.values())




class Fooling3DBatchSampler(data.Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of each batch (how many frames from the same video).
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """
        This will return indices of frames in a single video folder, ensuring batch contains only frames from that video.
        """
        for video_idx in range(len(self.dataset.video_frames_info)):
            frames_info = self.dataset.video_frames_info[video_idx]
            num_frames = len(frames_info)
            frame_idx_list = list(np.arange(num_frames))

            # # Shuffle the frame indices if shuffle is True
            # if self.shuffle:
            #     np.random.shuffle(frame_idx_list)

            # If frames count is not divisible by batch size, repeat the last frame
            if num_frames % self.batch_size != 0:
                num_repeat = self.batch_size - (num_frames % self.batch_size)
                frame_idx_list += [frame_idx_list[-1]] * num_repeat  # Add last frame to fill up batch

            # Yield frames in batches of batch_size
            for i in range(0, len(frame_idx_list), self.batch_size):
                batch_info = [frames_info[frame_idx] for frame_idx in frame_idx_list[i:i + self.batch_size]]
                yield batch_info

    def __len__(self):
        """
        The length of the sampler is the number of total batches in all videos.
        """
        total_batches = 0
        for frames_info in self.dataset.video_frames_info:
            total_batches += len(frames_info) // self.batch_size + (1 if len(frames_info) % self.batch_size != 0 else 0)
        return total_batches



class DistributedFooling3DBatchSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of each batch (how many frames from the same video).
            num_replicas (int): Total number of processes (GPUs) across all nodes.
            rank (int): Rank of the current process (GPU) in the group of workers.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()

    def __iter__(self):
        """
        This will return indices of frames in a single video folder, ensuring batch contains only frames from that video.
        Distributes the frames across different processes.
        """
        for video_idx in range(len(self.dataset.video_frames_info)):
            frames_info = self.dataset.video_frames_info[video_idx]
            num_frames = len(frames_info)
            frame_idx_list = list(np.arange(num_frames))

            # # Shuffle the frame indices if shuffle is True
            # if self.shuffle:
            #     np.random.shuffle(frame_idx_list)

            # If frames count is not divisible by batch size, repeat the last frame
            if num_frames % self.batch_size != 0:
                num_repeat = self.batch_size - (num_frames % self.batch_size)
                frame_idx_list += [frame_idx_list[-1]] * num_repeat  # Add last frame to fill up batch

            # Total number of batches across all replicas
            num_batches = len(frame_idx_list) // self.batch_size + (1 if len(frame_idx_list) % self.batch_size != 0 else 0)
            
            # Divide the dataset into chunks and ensure each rank gets its share
            # Find out how many batches each rank should process
            chunks_per_rank = num_batches // self.num_replicas
            remainder = num_batches % self.num_replicas
            start_idx = self.rank * chunks_per_rank + min(self.rank, remainder)
            end_idx = (self.rank + 1) * chunks_per_rank + min(self.rank + 1, remainder)
            
            # Generate the frames indices for the current process's portion of the data
            for i in range(start_idx, end_idx):
                batch_info = [frames_info[frame_idx] for frame_idx in frame_idx_list[i * self.batch_size:(i + 1) * self.batch_size]]
                yield batch_info

    def __len__(self):
        """
        The length of the sampler is the total number of batches divided across all processes.
        """
        total_batches = 0
        for frames_info in self.dataset.video_frames_info:
            total_batches += len(frames_info) // self.batch_size + (1 if len(frames_info) % self.batch_size != 0 else 0)
        
        # Divide the total batches by the number of processes
        return total_batches // self.num_replicas + (1 if total_batches % self.num_replicas > self.rank else 0)

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.lower().startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''), args=args)
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury")
        elif dataset_name.lower() == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass', args=args)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass', args=args)
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name.lower():
            new_dataset = KITTI(aug_params, split=dataset_name, args=args)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name.lower() == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params, args=args)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name.lower() == 'falling_things':
            new_dataset = FallingThings(aug_params, args=args)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.lower().startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif 'nerfstereo' in dataset_name.lower():
            new_dataset = NerfStereoDataset(aug_params, args=args, root='./datasets/NerfStereo', txt_root='./datasets/NerfStereo/../')
            logging.info(f"Adding {len(new_dataset)} samples from NerfStereoDataset")
        elif 'crestereo' in dataset_name.lower():
            new_dataset = CREStereoDataset(aug_params, args=args, txt_root='./datasets/CREStereo_dataset/../')
            logging.info(f"Adding {len(new_dataset)} samples from CREStereoDataset")
        elif dataset_name.lower() == 'fsd':
            new_dataset = FSDDataset(aug_params, args=args, txt_root='./datasets/FSD/')
            logging.info(f"Adding {len(new_dataset)} samples from FSDDataset")
        elif dataset_name.lower() == 'infstereo':
            new_dataset = InfStereoDataset(aug_params, args=args, root='./datasets/InfinigenStereo/release_full', caching=True)
            logging.info(f"Adding {len(new_dataset)} samples from InfStereoDataset")
        elif dataset_name.lower() == 'trans':
            new_dataset = Trans(aug_params, args=args)
            logging.info(f"Adding {len(new_dataset)} samples from Trans")
        elif dataset_name.lower() == 'fooling3d':
            new_dataset = Fooling3DDataset(aug_params, args=args, root='./datasets/Fooling3D')
            # print("+"*10, hasattr(args, 'enable_sampler') and args.enable_sampler)
            if hasattr(args, 'enable_sampler') and args.enable_sampler:
                # sampler = Fooling3DBatchSampler(new_dataset, args.batch_size)
                sampler = DistributedFooling3DBatchSampler(new_dataset, args.batch_size)
            logging.info(f"Adding {len(new_dataset)} samples from Fooling3DDataset")
            # TODO: Add Fooling3D dataset with only one sampler may cause conflict with other datasets
        else:
            raise Exception(f"Dataset {dataset_name} is not defined")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
    #     pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)
    train_loader = get_loader(train_dataset, args)
    train_loader.sampler.set_epoch(0)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader





def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Dataset sample visualizer")
    parser.add_argument("--root", default="datasets/FSD", help="Dataset root directory")
    parser.add_argument("--txt-root", default='./datasets/FSD/', help="Optional cached file-list root")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize")

    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    args = parser.parse_args()
    print("-" * 20)

    aug_params = None
    aug_params = {
                  'crop_size': args.image_size, 
                  'min_scale': args.spatial_scale[0], 
                  'max_scale': args.spatial_scale[1], 
                  'do_flip': False, 
                  'yjitter': not args.noyjitter,
                #   'yjitter': False,
                  }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    # dataset = FSDDataset(aug_params=aug_params, root=args.root, txt_root=args.txt_root)
    dataset = InfStereoDataset(aug_params=aug_params)
    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no samples")

    idx = max(0, min(args.index, len(dataset) - 1))
    metadata, left, right, flow, valid, _ = dataset[idx]
    print(idx, metadata)

    left_img = _tensor_to_image(left)
    right_img = _tensor_to_image(right)
    disparity = (-flow[0]).detach().cpu().numpy()
    mask = valid.squeeze().detach().cpu().numpy()
    print(mask.shape, disparity.shape, np.sum(mask), mask.max(), mask.min())
    # disparity = np.ma.masked_where(~mask, disparity)

    output_dir = Path("tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sample_{idx}"
    if isinstance(metadata, (list, tuple)) and metadata:
        base_name = Path(metadata[0]).stem or base_name

    plt.imsave(output_dir / f"{base_name}_left.png", left_img)
    plt.imsave(output_dir / f"{base_name}_right.png", right_img)
    plt.imsave(output_dir / f"{base_name}_disparity.png", np.ma.filled(disparity, np.nan), cmap="jet")
    frame_utils.writePFM(str(output_dir / f"{base_name}_disparity.pfm"), disparity)
    plt.imsave(output_dir / f"{base_name}_mask.png", mask, cmap="gray", vmin=0, vmax=1)


if __name__ == "__main__":
    main()