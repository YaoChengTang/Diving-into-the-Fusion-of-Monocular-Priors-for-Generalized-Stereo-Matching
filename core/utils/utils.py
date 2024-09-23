import os
import sys
import time
import shutil
import logging
import numpy as np

from scipy import interpolate
from datetime import datetime

import torch
import torch.nn.functional as F



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def hor_coords_grid(batch, ht, wd):
    # (batch,1,H,W)
    hor_coords = torch.arange(wd).float().repeat(batch, 1, ht, 1)
    return hor_coords


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)

def disparity_computation(params, slant=None, slant_norm=False, coords0=None):
    """
    args:
        params: (B,C,...), C is the type of parameters.
        coords0: (B,C,...), C is the number of coordinates' axis.
    """
    if slant is None or len(slant)==0 :
        offset = params
    elif slant=="slant" :
        # d = a*u + b*v + c
        B,H,W = coords0.shape[0], coords0.shape[-2], coords0.shape[-1]
        if slant_norm:
            norm_range = torch.Tensor([W,H])[None,:,None,None].float().to(coords0.device)
            offset = params[:,0] * coords0[:,0] / norm_range[:,0] + \
                     params[:,1] * coords0[:,1] / norm_range[:,1] + \
                     params[:,2]
        else:
            offset = params[:,0] * coords0[:,0] + \
                     params[:,1] * coords0[:,1] + \
                     params[:,2]
    elif slant=="slant_local":
        raise Exception("slant_local is not supported")
    else:
        raise Exception(f"{slant} is not supported")
    return offset


def rescale_modulation(itr, iters, modulation_alg, modulation_ratio):
    # we hope modulation has less effect at the first several iterations as the disp is unreliable and the lcoal LBP disp is unreliable
    if modulation_alg == "linear":
        ratio = modulation_ratio * itr / iters
    elif modulation_alg == "sigmoid":
        ratio = modulation_ratio * 1 / (1 + np.exp(-2 * (itr - 5)))
    else:
        raise Exception("Not supported modulation_alg: {}".format(modulation_alg))
    return ratio



NODE_RANK    = os.getenv('NODE_RANK', default=0)
LOCAL_RANK   = os.getenv("LOCAL_RANK", default=0)
LOG_ROOT     = os.getenv('LOG_ROOT', default="logs")
TB_ROOT      = os.getenv('TB_ROOT', default="runs")

class LoggerCommon:
    def __init__(self, name):
        self.name = name
        self.log_name = '{}-{}.log'.format(name, datetime.now().strftime("%y%m%d_%H%M%S"))
        self.log_path = os.path.join(LOG_ROOT, self.log_name)
        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            os.makedirs(LOG_ROOT, exist_ok=True)
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                handlers = [logging.FileHandler(self.log_path), 
                                            logging.StreamHandler()]
                            )
            self.logger = logging.getLogger(name)
            self.logger.addHandler(logging.FileHandler(self.log_path))
    
    def _set_handlers(self):
        # 清除之前的所有处理器
        self.logger.handlers.clear()
        
        # 设置文件和控制台处理器
        file_handler = logging.FileHandler(self.log_path)
        console_handler = logging.StreamHandler()

        # 设置处理器格式
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_log_path(self, new_log_path, name=None):
        # 删除旧日志文件（如果存在）
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        
        # 更新路径并重设处理器
        if name is not None:
            self.name = name
        self.log_name = '{}-{}.log'.format(self.name, datetime.now().strftime("%y%m%d_%H%M%S"))
        self.log_path = os.path.join(new_log_path, self.log_name)
        os.makedirs(new_log_path, exist_ok=True)
        self._set_handlers()

    
    def info(self, message):
        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            self.logger.info(message)
    
    def warning(self, message):
        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            self.logger.warning(message)
    
    def error(self, message):
        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            self.logger.error(message)
    
    def exception(self, message):
        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            self.logger.exception(message)
    
    def print_args(self, args):
        msg = ""
        args_dict = vars(args)
        max_arg_length = max(len(arg_name) for arg_name in args_dict.keys())
        for arg_name, arg_value in args_dict.items():
            arg_name_padded = arg_name.ljust(max_arg_length)
            msg += f"{arg_name_padded}: {arg_value}\r\n"
        self.info(msg)
        


from torch.utils.tensorboard import SummaryWriter

class LoggerTraining(LoggerCommon):

    SUM_FREQ = 100

    def __init__(self, name, model=None, scheduler=None):
        super(LoggerTraining, self).__init__(name)

        if int(LOCAL_RANK)==0 and int(NODE_RANK)==0:
            os.makedirs(TB_ROOT, exist_ok=True)

        self.model = model
        self.scheduler = scheduler
        self.silence = False
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=TB_ROOT)
    
    def set_training(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
    
    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/LoggerTraining.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        self.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=TB_ROOT)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/LoggerTraining.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % LoggerTraining.SUM_FREQ == LoggerTraining.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=TB_ROOT)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()



def init_directories(directories):
    if int(LOCAL_RANK)==0 and int(NODE_RANK)==0 :
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
def delete_directories_if_static(directories):
    if int(LOCAL_RANK)==0 and int(NODE_RANK)==0 :
        # 如果检测到文件大小有变化，终止删除操作
        if not is_any_folder_static(directories):
            print("File sizes are changing in one of the directories {}.".format(directories) + \
                  "No directories will be deleted.")
            return
        
        # 如果所有文件都静止，删除目录
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Directory {directory} deleted")

def get_file_sizes(directories):
    """返回多个目录中所有文件的大小字典"""
    file_sizes = {}
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    file_sizes[filepath] = os.path.getsize(filepath)
    return file_sizes

def is_any_folder_static(directories, check_interval=2):
    """检测所有文件是否静止（没有变化）"""
    # 获取所有文件初始大小
    initial_sizes = get_file_sizes(directories)
    time.sleep(check_interval)  # 等待一段时间，观察文件变化
    final_sizes = {filepath: os.path.getsize(filepath) for filepath in initial_sizes if os.path.exists(filepath)}
    
    # 如果文件大小一致，则所有文件静止
    return initial_sizes == final_sizes