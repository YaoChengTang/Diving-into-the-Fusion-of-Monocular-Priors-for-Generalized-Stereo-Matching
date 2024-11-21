import numpy as np
import random
import os
import sys
import subprocess
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression

sys.path.insert(0,'core')
sys.path.insert(0,'core/utils')

from core.utils import frame_utils
from core.utils import vis

gt_path = "/home/yao/Document/GeneralizedStereoMatching/datasets/Middlebury/MiddEval3/trainingF/Jadeplant/disp0GT.pfm"
pse_gt_path = "/data5/yao/runs/vis/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/middlebury_F-50000itr-RefineSigmoidPreMonoBatch48ConfDim/middlebury_f/MiddEval3/trainingF/Jadeplant/disp0GT-pred.pfm"
pred_path = "/data5/yao/tmp/Jadeplant-depthany.png"


gt, valid = frame_utils.readDispMiddlebury(gt_path)
pse_gt = frame_utils.readPFM(pse_gt_path)

pred = cv2.imread(pred_path, cv2.IMREAD_ANYDEPTH)
print(gt.shape, pred.shape, pse_gt.shape, gt.max())


class Args:
    def __init__(self, dataset):
        self.dataset = dataset
args = Args(dataset="middlebury_F")
viser = vis.Visualizer(root="/data5/yao/tmp/register", sv_root="/data5/yao/tmp/register", dataset="middlebury", scratch=True, args=args, logger=None)


Y = gt[valid].reshape((-1,1))
X = pred[valid].reshape((-1,1))
X = np.hstack([X, np.ones_like(X)])
print(X.shape, Y.shape, X.max(), Y.max())


model = LinearRegression(fit_intercept=False) 
model.fit(X, Y)


a, b = model.coef_[0][0], model.coef_[0][1]
reg_pred = pred * a + b
epe = np.abs(reg_pred[valid]-gt[valid]).mean()
bad3 = (epe>3).mean()
print(f"{a}, {b}: ", epe, bad3)

# vis1 = [{"name": "GT Disp", "img_list": [gt], "cmap": "jet"},
#         {"name": "PSE GT Disp", "img_list": [pse_gt], "cmap": "jet"},
#         {"name": "Pred Disp", "img_list": [pred], "cmap": "jet"},
#         {"name": "Reg Disp", 
#             "img_list": [reg_pred], 
#             "cmap": "jet",
#             "GT": [gt],
#             "error_map": True,},
#         ]
# viser.analyze(vis1, os.path.basename(gt_path), in_one_fig=True)



ransac = RANSACRegressor(estimator=LinearRegression(), max_trials=1000, min_samples=500, residual_threshold=1.0)
ransac.fit(X, Y)


a = ransac.estimator_.coef_[0][0]
b = ransac.estimator_.intercept_
reg_pred = pred * a + b
epe = np.abs(reg_pred[valid]-gt[valid]).mean()
bad3 = (epe>3).mean()
print(f"{a}, {b}: ", epe, bad3)

# vis1 = [{"name": "GT Disp", "img_list": [gt], "cmap": "jet"},
#         {"name": "PSE GT Disp", "img_list": [pse_gt], "cmap": "jet"},
#         {"name": "Pred Disp", "img_list": [pred], "cmap": "jet"},
#         {"name": "Reg Disp", 
#             "img_list": [reg_pred], 
#             "cmap": "jet",
#             "GT": [gt],
#             "error_map": True,},
#         ]
# viser.analyze(vis1, os.path.basename(gt_path), in_one_fig=True)