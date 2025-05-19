# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from core.utils import plane
from core.utils import frame_utils
from core.utils.ddp import get_loader
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
DATASET_ROOT = os.getenv('DATASET_ROOT')


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
            intrinsic = torch.from_numpy(np.array(intrinsic)).float() if intrinsic is not None else torch.from_numpy(np.eye(3)).float()
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


from torch.utils.data.distributed import DistributedSampler
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
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''), args=args)
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury")
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass', args=args)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass', args=args)
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI(aug_params, split=dataset_name, args=args)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params, args=args)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params, args=args)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif 'nerfstereo' in dataset_name:
            new_dataset = NerfStereoDataset(aug_params, args=args, root='./datasets/NerfStereo', txt_root='./datasets/NerfStereo/../')
            logging.info(f"Adding {len(new_dataset)} samples from NerfStereoDataset")
        elif 'crestereo' in dataset_name:
            new_dataset = CREStereoDataset(aug_params, args=args, txt_root='./datasets/CREStereo_dataset/../')
            logging.info(f"Adding {len(new_dataset)} samples from CREStereoDataset")
        elif dataset_name == 'Trans':
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
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
    #     pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)
    train_loader = get_loader(train_dataset, args)
    train_loader.sampler.set_epoch(0)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

