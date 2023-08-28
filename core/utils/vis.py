import os
import sys
from utils.frame_utils import writeDispMiddlebury, writeDispKITTI, write_gen

class Visualizer:
    def __init__(self, root, sv_root, dataset=None, scratch=True):
        self.root = root.rstrip("/")
        self.sv_root = sv_root.rstrip("/")
        self.dataset = dataset
        self.scratch = scratch
        print("saving to {}".format(self.sv_root))

    def save_pred_vis(self, flow_pr, imageGT_file):
        assert self.root in imageGT_file, "{} not in {}".format(self.root, imageGT_file)

        # create saving path, /xxx/disp0GT.pfm -> /xxx/disp0GT-pred.pfm
        sv_path = imageGT_file.replace(self.root, self.sv_root)
        pre,lat = os.path.splitext(sv_path)
        sv_path = pre + "-pred" + lat
        if not self.scratch and os.path.exists(sv_path):
            print("{} exists".format(sv_path))
            return True

        # build directory
        sv_dir = os.path.dirname(sv_path)
        if not os.path.exists(sv_dir) :
            os.makedirs(sv_dir)

        # write prediction
        if self.dataset.lower()=="middlebury" :
            writeDispMiddlebury(sv_path, flow_pr)
        elif self.dataset.lower()=="kitti2015" :
            writeDispKITTI(sv_path, flow_pr)
        elif self.dataset.lower()=="eth3d" :
            write_gen(sv_path, flow_pr)
        else:
            raise Exception("such daatset is not supported: {}".format(dataset))
        return True
        