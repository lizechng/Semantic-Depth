import os
from PIL import Image
import numpy as np
import math
import torch


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    valid_mask = (gt > 0) & (gt < 80)
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    mse = (gt - pred) ** 2
    rmse = torch.sqrt(mse.float().mean())
    mae = torch.sqrt(mse.float()).mean()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = (torch.sqrt(mse.float()) / gt).float().mean()
    sq_rel = (((gt - pred) ** 2) / gt).float().mean()

    thresh = torch.max((gt / pred), (pred / gt))
    # thresh = (gt + torch.sqrt(mse.float())) / gt
    d1 = float((thresh < 1.25).float().mean())
    d2 = float((thresh < 1.25 ** 2).float().mean())
    d3 = float((thresh < 1.25 ** 3).float().mean())

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae

gt = '/media/personal_data/lizc/DepthCompletion/DepthTest/lidar/val/'
ours_ = './result_255/test/gray/depth/' # train, test
DORN_radar = '/media/personal_data/lizc/DepthCompletion/DORN_radar/result/infer/val/gray/' # train, val

pred_path = DORN_radar

files = os.listdir(pred_path)
files = sorted(files)
_abs_rel, _sq_rel, _rmse, _rmse_log, _d1, _d2, _d3, _mae = 0, 0, 0, 0, 0, 0, 0, 0
for file in files:
    # Image.open(os.path.join(pred_path, file)).show()
    # Image.open(os.path.join(gt, file)).show()
    pred = np.asarray(Image.open(os.path.join(pred_path, file)))
    target = np.asarray(Image.open(os.path.join(gt, file)))
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    assert pred.shape == target.shape
    abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae = compute_errors(target, pred)
    if np.isnan(d1):
        mask = target > 0
        t = target[mask]
        print(t.shape)
        continue

    _abs_rel += abs_rel
    _sq_rel += sq_rel
    _rmse += rmse
    _rmse_log += rmse_log
    _d1 += d1
    _d2 += d2
    _d3 += d3
    _mae += mae

print('abs_rel: ', _abs_rel / len(files))
print('sq_rel: ', _sq_rel / len(files))
print('rmse: ', _rmse / len(files))
print('rmse_log: ', _rmse_log / len(files))
print('d1: ', _d1 / len(files))
print('d2: ', _d2 / len(files))
print('d3: ', _d3 / len(files))
print('mae: ', _mae / len(files))


