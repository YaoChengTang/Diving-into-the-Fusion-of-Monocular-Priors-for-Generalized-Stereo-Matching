import os
import sys
import numpy as np
sys.path.append('core')

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from core.utils import frame_utils
from core.utils import vis


disp_path = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow/flyingthings3d/disparity/TRAIN/A/0717/left/0006.pfm"
left_path = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow/flyingthings3d/frames_cleanpass/TRAIN/A/0717/left/0006.png"
sv_path   = "./tmp.png"

disp = np.array(frame_utils.readPFM(disp_path))
img0 = np.array(Image.open(left_path))
H,W = disp.shape

disp = torch.from_numpy(disp).view((1,1,H,W))
img0 = torch.from_numpy(img0).permute((2,0,1)).view((1,3,H,W))

u,v = torch.arange(W), torch.arange(H)
grid_u, grid_v  = torch.meshgrid(u, v, indexing="xy")
# print(grid_u.shape, grid_v.shape)
# print(grid_u[0:2,:10], grid_v[0:2, :10], sep="\r\n")
# grid_u = grid_u.view((1,1,H,W))
# grid_v = grid_v.view((1,1,H,W))
# pos = torch.cat([grid_u,grid_v,disp],dim=1)
pos = disp
pos_len = pos.shape[1]

patch_pos = F.unfold(pos, kernel_size=4, dilation=1, padding=0, stride=4)
patch_pos = patch_pos.view((1,pos_len,16,H//4,W//4))
src = patch_pos.view((1,pos_len,16,1,H//4,W//4)).tile((1,1,1,16,1,1))
tar = patch_pos.view((1,pos_len,1,16,H//4,W//4)).tile((1,1,16,1,1,1))
dist = torch.sqrt(torch.square(src-tar).sum(dim=1))
connect = (dist<3).float()
for _ in range(10):
    connect = torch.einsum('bijhw,bjkhw->bikhw', connect, connect)
connect = (connect>0).sum(dim=2)
mask = connect - torch.amax(connect,dim=1).view((1,1,H//4,W//4))
mask = mask >= -0.0001
# print(disp.shape, img0.shape, patch_pos.shape, dist.shape, connect.shape, mask.shape)
# test_v, test_u = 100,100
# torch.set_printoptions(precision=2)
# print(src[0,:,0,:,test_v, test_u], tar[0,:,0,:,test_v, test_u], patch_pos[0,:,:,test_v, test_u], dist[0,:,:,test_v, test_u], sep="\r\n")
# print(connect[0,:,test_v, test_u], mask[0,:,test_v, test_u], sep="\r\n")


disp = disp.view((H,W)).cpu().data.numpy()
img0 = img0.view((3,H,W)).permute((1,2,0)).cpu().data.numpy()
# patch_disp = patch_disp[0,0,...].cpu().data.numpy()

# vis.show_imgs([{"img":img0, "title":"Left Image", },
#                {"img":disp, "title":"GT Disparity", "cmap":'jet', },
#                {"img":patch_disp, "title":"GT Patch Disparity", "cmap":'jet', },
#             ], 
#             sv_img=True, save2where=sv_path, if_inter=False, 
#             fontsize=20, szWidth=10, szHeight=5, group=3)