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



def resize_tensor(tensor, target_size=512, ratio=16):
    # Get the input tensor shape (B, C, H, W)
    _, _, H, W = tensor.shape
    
    # Calculate the longer side between H and W
    if H > W:
        new_H = target_size
        new_W = int(W * (target_size / H))
    else:
        new_W = target_size
        new_H = int(H * (target_size / W))
    
    # Round up new_W and new_H to be divisible by ratio
    new_W = (np.ceil(new_W / ratio) * ratio).astype(int)
    new_H = (np.ceil(new_H / ratio) * ratio).astype(int)

    # Resize using interpolate
    resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bicubic', align_corners=False)
    
    return resized_tensor


def resize_to_quarter(tensor, original_size, ratio):
    # Scale down to 1/ratio of the original size
    quarter_H = original_size[0] // ratio
    quarter_W = original_size[1] // ratio
    
    # Resize using interpolate
    resized_tensor = F.interpolate(tensor, size=(quarter_H, quarter_W), mode='bilinear', align_corners=False)
    
    return resized_tensor


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
    
    def pad_intrinsics(self, intrinsic):
        intrinsic[..., 2] += self._pad[0]
        intrinsic[..., 3] += self._pad[2]
        return intrinsic

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


def sv_intermediate_results(data, name, sv_path):
    try:
        sv_path = os.path.join(sv_path, "data")
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        
        data_numpy = data.cpu().data.numpy()
        np.save(os.path.join(sv_path, name+".npy"), data_numpy)
        # print("saving to {}".format( os.path.join(sv_path, name+".npy") ))
    except Exception as err:
        raise Exception(err, data.shape, name, sv_path)

def load_intermediate_results(name, sv_path):
    sv_path = os.path.join(sv_path, "data")
    data = np.load(os.path.join(sv_path, name+".npy"))
    return data


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
    def __init__(self, name, log_root='', local_rank=0, node_rank=0):
        """
        Initialize a logger.
        
        Args:
            name (str): Logger name.
            log_root (str): Directory to save log files.
            local_rank (int): Local process rank in distributed training.
            node_rank (int): Node rank in distributed training.
        """
        self.name = name
        self.log_root = log_root if log_root else LOG_ROOT
        self.local_rank = int(local_rank)
        self.node_rank = int(node_rank)
        self.log_name = f"{self.name}-{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(self.log_root, self.log_name)
        self.logger = logging.getLogger(name)
        self.logger.propagate = False  # Prevent logs from going to root
        
        # Only add handlers for rank 0 to avoid duplicate logs
        if self.local_rank == 0 and self.node_rank == 0:
            if len(self.logger.handlers) == 0:  # Avoid adding handlers multiple times
                os.makedirs(self.log_root, exist_ok=True)
                self._set_handlers()
                # print("Handlers set for logger:", self.name)
        # print("-"*30, self.name, "Logger initialized. Log path:", self.log_path)

    def _set_handlers(self):
        """Clear old handlers and add new file and console handlers."""
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)

        # Log message format
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_log_path(self, new_log_root, name=None):
        """
        Change the log directory and reset handlers.

        Args:
            new_log_root (str): New directory to save logs.
            name (str, optional): New logger name.
        """
        if name:
            self.name = name
        self.log_root = new_log_root
        self.log_name = f"{self.name}-{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(self.log_root, self.log_name)
        os.makedirs(self.log_root, exist_ok=True)
        if self.local_rank == 0 and self.node_rank == 0:
            self._set_handlers()

    # Logging interfaces
    def info(self, message):
        if self.local_rank == 0 and self.node_rank == 0:
            self.logger.info(message)

    def warning(self, message):
        if self.local_rank == 0 and self.node_rank == 0:
            self.logger.warning(message)

    def error(self, message):
        if self.local_rank == 0 and self.node_rank == 0:
            self.logger.error(message)

    def exception(self, message):
        if self.local_rank == 0 and self.node_rank == 0:
            self.logger.exception(message)

    def print_args(self, args):
        """
        Print the arguments in a neat formatted way.

        Args:
            args: Object with attributes (e.g., argparse.Namespace)
        """
        args_dict = vars(args)
        max_len = max(len(k) for k in args_dict)
        msg = "\n".join(f"{k.ljust(max_len)}: {v}" for k, v in args_dict.items())
        self.info(msg)

        

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class LoggerTraining(LoggerCommon):

    SUM_FREQ = 100

    def __init__(self, logging_name, model=None, scheduler=None, use_wandb=False, exp_name="default", project_name="BD"):
        super(LoggerTraining, self).__init__(logging_name)

        if int(LOCAL_RANK) == 0 and int(NODE_RANK) == 0:
            os.makedirs(TB_ROOT, exist_ok=True)

        self.model = model
        self.scheduler = scheduler
        self.silence = False
        self.total_steps = 0
        self.running_loss = {}

        # Decide whether to use wandb or tensorboard
        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        if self.use_wandb:
            if wandb.run is None:  # Initialize if not already done
                wandb.init(project=project_name, name=exp_name, dir=TB_ROOT)
            self.writer = None
            self.info(f"Using wandb with project name: {project_name}, run name: {exp_name}.")
        else:
            self.writer = SummaryWriter(log_dir=TB_ROOT)
            self.info(f"Using TensorBoard with log dir {TB_ROOT}.")
    
    def set_training(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
    
    def _print_training_status(self):
        metrics_data = {k: self.running_loss[k] / LoggerTraining.SUM_FREQ
                        for k in sorted(self.running_loss.keys())}

        training_str = "[{:6d}, {:10.7f}]".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = " | ".join([f"{k}: {v:10.4f}" for k, v in metrics_data.items() if k != "lr"])
        
        # Print keys and values
        self.info(f"Training Metrics {training_str}: {metrics_str}")

        if self.use_wandb:
            wandb.log(metrics_data, step=self.total_steps)
        else:
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=TB_ROOT)
            for k, v in metrics_data.items():
                self.writer.add_scalar(k, v, self.total_steps)

        # Reset
        self.running_loss = {}

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        if self.total_steps % LoggerTraining.SUM_FREQ == LoggerTraining.SUM_FREQ - 1:
            self._print_training_status()
    
    def add_scalar(self, key, value, step=None):
        if self.use_wandb:
            wandb.log({key: value}, step=step)
        else:
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=TB_ROOT)
            self.writer.add_scalar(key, value, step)

    def write_dict(self, results):
        if self.use_wandb:
            wandb.log(results, step=self.total_steps)
        else:
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=TB_ROOT)
            for key in results:
                self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.use_wandb:
            wandb.finish()
        elif self.writer:
            self.writer.close()


def init_directories(directories):
    if int(LOCAL_RANK) == 0 and int(NODE_RANK) == 0:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    

def delete_directories_if_static(directories):
    if int(LOCAL_RANK) == 0 and int(NODE_RANK) == 0:
        # If file sizes are changing, stop the deletion process
        if not is_any_folder_static(directories):
            print("File sizes are changing in one of the directories {}.".format(directories) +
                  "No directories will be deleted.")
            return
        
        # If all files are static, delete directories
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Directory {directory} deleted")


def get_file_sizes(directories):
    """Return a dictionary of all file sizes in multiple directories"""
    file_sizes = {}
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    file_sizes[filepath] = os.path.getsize(filepath)
    return file_sizes


def is_any_folder_static(directories, check_interval=2):
    """Check whether all files are static (no changes)"""
    # Get initial sizes of all files
    initial_sizes = get_file_sizes(directories)
    time.sleep(check_interval)  # Wait a while to observe file changes
    final_sizes = {filepath: os.path.getsize(filepath) for filepath in initial_sizes if os.path.exists(filepath)}
    
    # Return True if file sizes are the same, i.e., all files are static
    return initial_sizes == final_sizes
