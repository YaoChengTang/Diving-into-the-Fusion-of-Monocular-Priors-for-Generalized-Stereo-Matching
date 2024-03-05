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
        cab: B,6,H,W; (disparity, a, b, g_uu, g_vv, g_uv)
    """
    # "slant": get a*u + b*v - d + c = 0 through least squares
    # "slant_local": a*(u-u_p) + b*(b-b_p) - (d-d_p) = 0
    B,C,L,H,W = chs_coord.shape
    chs_coord = chs_coord.flatten(-2,-1).transpose(-2,-1)                                       # (B,C,H*W,patch_size*patch_size)
    u_coord = chs_coord[:,0]
    v_coord = chs_coord[:,1]
    d_coord = chs_coord[:,2]
    A = torch.stack((torch.ones_like(u_coord), u_coord, v_coord, 
                    u_coord*u_coord/2, v_coord*v_coord/2, u_coord*v_coord), dim=3)              # (B,H*W,patch_size*patch_size,6)

    # print(chs_coord.shape, A.shape, d_coord.shape)
    cab = torch.linalg.lstsq(A, d_coord).solution   # B,H*W,C
    cab = cab.transpose(1,2).view((-1,6,H,W))

    # # A(B,N,P,C) X(B,N,C) Y(B,N,P)
    # # print("-"*10, A.shape, d_coord.shape, abc.shape)
    # left_top = torch.einsum('aijk,aikh->aijh', A.transpose(-1,-2), A)                         # (B,N,C,C)
    # right_top = -torch.einsum('aijk,aikh->aijh', A.transpose(-1,-2), d_coord.unsqueeze(-1))   # (B,N,C,1)
    # left_bottom = right_top.transpose(-1,-2)                                                  # (B,N,1,C)
    # right_bottom = d_coord.square().sum(dim=-1,keepdim=True).unsqueeze(-1)                    # (B,N,1,1)
    # top = torch.cat([left_top,right_top], dim=3)
    # bottom = torch.cat([left_bottom,right_bottom], dim=3)
    # B = torch.cat([top,bottom], dim=2)
    # L, V = torch.linalg.eig(B)
    # print(L, V.shape)

    return cab

def extract_plane(disp,slant="slant", slant_norm=False, patch_size=4,thold=3,vis=False):
    """
    disp: B,1,H,W;
    return:
        cab: B,6,H,W; (disparity, a, b, g_uu, g_vv, g_uv)
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
    cab = get_plane_lstsq(chs_coord, slant, patch_coord)
    
    if vis:
        return cab, mask
    return cab

def predict_disp(cab, uv_coord, patch_size, mul_last=False):
    """
    cab: B,6,H,W; (disparity, a, b, g_uu, g_vv, g_uv)
    uv_coord: B,2,patch_size*patch_size,H,W;
    """
    u_coord = uv_coord[:,0]
    v_coord = uv_coord[:,1]
    A = torch.stack((torch.ones_like(u_coord), u_coord, v_coord, 
                    u_coord*u_coord/2, v_coord*v_coord/2, u_coord*v_coord), dim=1)      # (B,6,patch_size*patch_size,H,W)
    d_coord = (A * cab.unsqueeze(dim=2)).sum(dim=1)
    if mul_last:
        d_coord *= patch_size
    # print(d_coord.shape)
    return d_coord

def compute_curvature(cab):
    """
    cab: B,6,H,W; (disparity, a, b, g_uu, g_vv, g_uv)

    """
    B,C,H,W = cab.shape
    hessian = torch.stack([cab[0,-3], cab[0,-1], cab[0,-1], cab[0,-2]],dim=-1).reshape(H,W,2,2)
    eigen_val, eigen_vec = torch.linalg.eigh(hessian)
    Gaussian_cur = eigen_val[...,0] * eigen_val[...,1]
    mean_cur     = (eigen_val[...,0] + eigen_val[...,1]) / 2

    Gaussian_cur = Gaussian_cur.abs()
    mean_cur = mean_cur.abs()

    Gaussian_cur[Gaussian_cur>0.03] = 0
    mean_cur[mean_cur>0.01] = 0

    Gaussian_cur = (Gaussian_cur - Gaussian_cur.min()) / (Gaussian_cur.max()-Gaussian_cur.min())
    mean_cur = (mean_cur - mean_cur.min()) / (mean_cur.max()-mean_cur.min())

    # print(Gaussian_cur[120, 170:180], mean_cur[120, 170:180], cab[0,-3:, 120, 170:180], sep="\r\n")

    return Gaussian_cur, mean_cur


if __name__ == '__main__':
    # slant = "slant"
    slant = "slant_local"
    # slant_norm = True
    slant_norm = False
    patch_size = 4
    root = "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
    disp_path = root+"/flyingthings3d/disparity/TRAIN/A/0717/left/0006.pfm"
    left_path = root+"/flyingthings3d/frames_cleanpass/TRAIN/A/0717/left/0006.png"
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
    # (B,6,H,W) ~ [disparity, u_coord, v_coord, g_uu, g_vv, g_uv]
    cab, mask = extract_plane(disp, 
                        slant=slant, slant_norm=slant_norm,
                        patch_size=patch_size, thold=3, vis=True)
    # print(cab.shape)

    uv_coord = get_pos(H,W, slant=slant, slant_norm=slant_norm, patch_size=patch_size)
    patch_uv_coord = convert2patch(uv_coord, patch_size=patch_size)
    d_coord = predict_disp(cab, patch_uv_coord, patch_size=patch_size, mul_last=True)

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
    print("cost time: {}".format(end_time-start_time), cab.shape)

    disp = disp.squeeze(0).squeeze(0).cpu().data.numpy()
    img0 = img0.squeeze(0).permute((1,2,0)).cpu().data.numpy()
    patch_disp = patch_disp[0,0,0,...].cpu().data.numpy()
    rec_disp = rec_disp[0,0,...].cpu().data.numpy()
    rec_mask = rec_mask[0,0,...].cpu().data.numpy()

    error_map = np.abs(rec_disp-disp)
    color_error_map = vis.colorize_error_map(error_map)

    # normals
    degree = torch.atan(cab[0,1] / cab[0,2])

    # curvatures
    Gaussian_cur, mean_cur = compute_curvature(cab)
    print("-"*10, Gaussian_cur.min(), Gaussian_cur.max(), Gaussian_cur.mean(), Gaussian_cur.median())
    print("-"*10, mean_cur.min(), mean_cur.max(), mean_cur.mean(), mean_cur.median())

    atom_dict = [{"img":img0, "title":"Left Image", },
                {"img":disp, "title":"GT Disparity", "cmap":'jet', },
                {"img":patch_disp, "title":"GT Patch Disparity", "cmap":'jet', },
                {"img":rec_disp, "title":"GT recover Disparity", "cmap":'jet', },
                {"img":rec_mask, "title":"rec_mask", "cmap": "gray"},
                {"img":color_error_map, "title":"color_error_map", },
                {"img":degree, "title":"GT ab", "cmap":'jet', },
                {"img":cab[0,0], "title":"GT c", "cmap":'jet', },

                {"img":Gaussian_cur.abs(), "title":"Gaussian curvature", "cmap":'jet', },
                {"img":mean_cur.abs(), "title":"mean curvature", "cmap":'jet', },
                ]
    
    if slant=="slant_local":
        d_p = cab[0,0]
        error_map = np.abs(d_p-patch_disp)
        color_error_map = vis.colorize_error_map(error_map)
        tmp_dict = [{"img":d_p, "title":"GT Disparity of Plane", "cmap":'jet', },
                    {"img":color_error_map, "title":"color_error_map of Plane", },]
        atom_dict += tmp_dict

    vis.show_imgs(atom_dict, 
                sv_img=True, save2where=sv_path, if_inter=False, 
                fontsize=20, szWidth=10, szHeight=5, group=2)