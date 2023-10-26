import os
import sys
import time
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

import frame_utils
import vis


def get_pos(H,W,disp=None,slant="slant",slant_norm=False,patch_size=None,device=None):
    if slant=="slant":
        u,v = torch.arange(W,device=device), torch.arange(H,device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
        if slant_norm:
            grid_u = grid_u/W
            grid_v = grid_v/H
    elif slant=="slant_local":
        assert H%patch_size==0 and W%patch_size==0
        if not slant_norm:
            u = torch.arange(-patch_size/2+0.5, patch_size/2-0.5 + 1, step=1, device=device)
            v = torch.arange(-patch_size/2+0.5, patch_size/2-0.5 + 1, step=1, device=device)
        else:
            # restrict into (-1,1)
            u = torch.arange(-1+1/patch_size, 1, step=2/patch_size, device=device)
            v = torch.arange(-1+1/patch_size, 1, step=2/patch_size, device=device)
        # print(u,v,sep="\r\n")
        u = u.tile((W//patch_size))
        v = v.tile((H//patch_size))
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    # print(grid_u.shape, grid_v.shape)
    # print(grid_u[0:2,:10], grid_v[0:2, :10], sep="\r\n")
    grid_u = grid_u.view((1,1,H,W))
    grid_v = grid_v.view((1,1,H,W))
    if disp is not None:
        pos = torch.cat([grid_u,grid_v,disp],dim=1)
    else:
        pos = torch.cat([grid_u,grid_v],dim=1)
    return pos.float()

def convert2patch(data, patch_size, div_last=False):
    """ 
    data: B,C,H,W;
    """
    B,C,H,W = data.shape
    assert H%patch_size==0 and W%patch_size==0
    patch_data = F.unfold(data, kernel_size=patch_size, dilation=1, padding=0, stride=patch_size)
    patch_data = patch_data.view((-1,C,patch_size*patch_size,H//patch_size,W//patch_size))
    if div_last:
        patch_data[:,-1] /= patch_size
    return patch_data

def intra_dist4patch(patch_data, patch_size):
    """
    patch_data: B,C,patch_size*patch_size,H,W
    """
    src = patch_data.unsqueeze(3).tile((1,1,1,patch_size*patch_size,1,1))
    tar = patch_data.unsqueeze(2).tile((1,1,patch_size*patch_size,1,1,1))
    dist = torch.sqrt(torch.square(src-tar).sum(dim=1))
    return dist

def get_adjacent_matrix(dist,patch_size,thold=3):
    connect = (dist<thold).float()
    max_loop = int(np.ceil(np.log2(patch_size*patch_size)))
    for _ in range(max_loop):
        connect = torch.einsum('bijhw,bjkhw->bikhw', connect, connect)
        connect = (connect>0).float()
    connect = (connect>0).sum(dim=2)
    return connect

def reduce_noise(patch_coord, mask):
    """
    patch_coord: B,C,patch_size*patch_size,H,W;
    mask: B,patch_size*patch_size,H,W;
    """
    # replace the other clique with center point of the largest clique
    center_coord = (patch_coord*mask.unsqueeze(1)).sum(dim=2) / mask.sum(dim=1)
    chs_coord = patch_coord*mask.unsqueeze(1) + (~mask.unsqueeze(1)) * center_coord.unsqueeze(2)
    # print(mask.shape, coord.shape, patch_coord.shape, chs_coord.shape)
    return chs_coord

# def abs2relative(patch_coord):
#     """
#     patch_coord: B,C,patch_size*patch_size,H,W;
#     """
#     center_patch_coord = patch_coord.mean(dim=2)
#     rel_patch_coord = patch_coord - center_patch_coord.unsqueeze(2)
#     return rel_patch_coord, center_patch_coord

def get_plane_lstsq(chs_coord, slant, patch_coord=None):
    """
    chs_coord: B,C,patch_size*patch_size,H,W;
    mask: B,patch_size*patch_size,H,W;
    return:
        abc: B,H*W,patch_size*patch_size;
    """
    # "slant": get a*u + b*v - d + c = 0 through least squares
    # "slant_local": a*(u-u_p) + b*(b-b_p) - (d-d_p) = 0
    B,C,L,H,W = chs_coord.shape
    chs_coord = chs_coord.flatten(-2,-1).transpose(-2,-1)
    u_coord = chs_coord[:,0]
    v_coord = chs_coord[:,1]
    d_coord = chs_coord[:,2]
    A = torch.stack((u_coord, v_coord, torch.ones_like(u_coord)), dim=3)
    # print(chs_coord.shape, A.shape, d_coord.shape)
    abc = torch.linalg.lstsq(A, d_coord).solution   # B,H*W,C
    abc = abc.transpose(1,2).view((-1,3,H,W))

    return abc

def extract_plane(disp,slant="slant", slant_norm=False, patch_size=4,thold=3,vis=False):
    """
    disp: B,1,H,W;
    """
    # cluster through nearest search
    patch_pos = convert2patch(disp, patch_size=patch_size)
    dist = intra_dist4patch(patch_pos, patch_size=patch_size)
    connect = get_adjacent_matrix(dist, patch_size=patch_size, thold=thold)

    # get the largest clique
    mask = connect - torch.amax(connect,dim=1).unsqueeze(1)
    mask = mask >= -0.0001
    # print((mask==0).sum(), (mask>0.5).sum(), mask.size())
    # print(disp[0,0,8:12,0:4], patch_pos[0,0,:,2,0], dist[0,:,:,2,0], connect[0,:,2,0], mask[0,:,2,0], sep="\r\n")

    # get the 3d coordinate (u,v,d) of each point
    B,_,H,W = disp.shape
    coord = get_pos(H,W,disp=disp,slant=slant,slant_norm=slant_norm,patch_size=patch_size)
    patch_coord = convert2patch(coord, patch_size=patch_size, div_last=True)

    # replace the other clique with center point of the largest clique
    chs_coord = reduce_noise(patch_coord, mask)
    # print(coord[0,:,400:404,400:404], patch_coord[0,:,:,100,100], chs_coord[0,:,:,100,100], sep="\r\n")

    # "slant": get a*u + b*v - d + c = 0 through least squares
    # "slant_local": a*(u-u_p) + b*(b-b_p) - (d-d_p) = 0
    abc = get_plane_lstsq(chs_coord, slant, patch_coord)
    
    if vis:
        return abc, mask
    return abc

def predict_disp(abc, uv_coord, patch_size, mul_last=False):
    """
    abc: B,3,H,W;
    uv_coord: B,2,patch_size*patch_size,H,W;
    """
    u_coord = uv_coord[:,0]
    v_coord = uv_coord[:,1]
    A = torch.stack((u_coord, v_coord, torch.ones_like(u_coord)), dim=1)
    d_coord = (A * abc.unsqueeze(dim=2)).sum(dim=1)
    if mul_last:
        d_coord *= patch_size
    # print(d_coord.shape)
    return d_coord


if __name__ == '__main__':
    slant = "slant"
    # slant = "slant_local"
    slant_norm = True
    # slant_norm = False
    patch_size = 4
    disp_path = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow/flyingthings3d/disparity/TRAIN/A/0717/left/0006.pfm"
    left_path = "/horizon-bucket/BasicAlgorithm/Users/chengtang.yao/Sceneflow/flyingthings3d/frames_cleanpass/TRAIN/A/0717/left/0006.png"
    sv_path   = "./tmp.png"

    img0 = np.array(Image.open(left_path))
    disp = np.array(frame_utils.readPFM(disp_path))
    # disp = np.zeros((20,20))
    # disp[9:] = 10
    H,W = disp.shape

    start_time = time.time()
    disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)
    img0 = torch.from_numpy(img0).permute((2,0,1)).unsqueeze(0)

    # extract planes a*u + b*v - d + c = 0
    abc, mask = extract_plane(disp, 
                        slant=slant, slant_norm=slant_norm,
                        patch_size=patch_size, thold=3, vis=True)
    # print(abc.shape)

    uv_coord = get_pos(H,W, slant=slant, slant_norm=slant_norm, patch_size=patch_size)
    patch_uv_coord = convert2patch(uv_coord, patch_size=patch_size)
    d_coord = predict_disp(abc, patch_uv_coord, patch_size=patch_size, mul_last=True)

    patch_disp = convert2patch(disp, patch_size=patch_size, div_last=True)
    rec_disp = F.fold(d_coord.flatten(-2,-1), disp.shape[-2:], kernel_size=patch_size, stride=patch_size).view(1,1,H,W)
    rec_mask = F.fold(mask.flatten(-2,-1).float(), disp.shape[-2:], kernel_size=patch_size, stride=patch_size).view(1,1,H,W).bool()
    # print(rec_disp.shape, patch_disp.shape, disp.shape[-2:])

    # print(disp.shape, img0.shape, patch_pos.shape, dist.shape, connect.shape, mask.shape)
    # test_v, test_u = 100,100
    # torch.set_printoptions(precision=2)
    # print(src[0,:,0,:,test_v, test_u], tar[0,:,0,:,test_v, test_u], patch_pos[0,:,:,test_v, test_u], dist[0,:,:,test_v, test_u], sep="\r\n")
    # print(connect[0,:,test_v, test_u], mask[0,:,test_v, test_u], sep="\r\n")

    end_time = time.time()
    print("cost time: {}".format(end_time-start_time))

    disp = disp.squeeze(0).squeeze(0).cpu().data.numpy()
    img0 = img0.squeeze(0).permute((1,2,0)).cpu().data.numpy()
    patch_disp = patch_disp[0,0,0,...].cpu().data.numpy()
    rec_disp = rec_disp[0,0,...].cpu().data.numpy()
    rec_mask = rec_mask[0,0,...].cpu().data.numpy()

    error_map = np.abs(rec_disp-disp)
    color_error_map = vis.colorize_error_map(error_map)

    atom_dict = [{"img":img0, "title":"Left Image", },
                {"img":disp, "title":"GT Disparity", "cmap":'jet', },
                {"img":patch_disp, "title":"GT Patch Disparity", "cmap":'jet', },
                {"img":rec_disp, "title":"GT recover Disparity", "cmap":'jet', },
                {"img":rec_mask, "title":"rec_mask", "cmap": "gray"},
                {"img":color_error_map, "title":"color_error_map", },
                ]
    
    if slant=="slant_local":
        d_p = abc[0,-1]
        error_map = np.abs(d_p-patch_disp)
        color_error_map = vis.colorize_error_map(error_map)
        tmp_dict = [{"img":d_p, "title":"GT Disparity of Plane", "cmap":'jet', },
                    {"img":color_error_map, "title":"color_error_map of Plane", },]
        atom_dict += tmp_dict

    vis.show_imgs(atom_dict, 
                sv_img=True, save2where=sv_path, if_inter=False, 
                fontsize=20, szWidth=10, szHeight=5, group=2)