# Dataset
import Datasets
from Datasets.dataloader import Dataset_loader
import numpy as np
from Models.sgd import semantic_depth_net
from torch.utils.data import DataLoader
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def colored_depthmap(depth, d_min=None, d_max=None):
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]


model = semantic_depth_net().cuda()
best_file_name = f'../img_seg_edge/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
                 f'checkpoint_model_epoch_149.pth.tar'
# best_file_name = f'../ImgSegEdge_255/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge1_patience7_num_samplesNone_multiFalse/' \
#                  f'checkpoint_model_epoch_149.pth.tar'
# best_file_name = f'../ImgSegEdge_255/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.07_wcls0.07_wdepth0.1_wseg5_wedge5_patience7_num_samplesNone_multiFalse/' \
#                  f'checkpoint_model_epoch_149.pth.tar'
print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))

dpath = '/media/personal_data/lizc/DepthCompletion/DepthTest/'
# dpath = '../DepthTrainData_352x1000'
dataset = Datasets.define_dataset('zhoushan', dpath, 'rgb')
dataset.prepare_dataset()
train_loader = Dataset_loader(None, dataset.val_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)

Data = DataLoader(train_loader, batch_size=1)
_abs_rel, _sq_rel, _rmse, _rmse_log, _d1, _d2, _d3, _mae = 0, 0, 0, 0, 0, 0, 0, 0
trainbar = tqdm(total=len(Data))
for i, (input, lidarGt, segGt) in enumerate(Data):
    img = np.asarray(input[0, 2:, :, :])
    img = img.transpose(1, 2, 0)
    # Image.fromarray(np.uint8(img * 255)).show()
    img = Image.fromarray(np.uint8(img * 255))
    img = np.asarray(img)
    input = input.cuda()

    res_coarse, res_cls, res_depth, seg, radar_points = model(input)

    seg_map = torch.argmax(seg, 1)[:, None]

    res_depth = res_depth[0, 0, :, :] * 256
    res_coarse = res_coarse[0, 0, :, :]
    res_cls = res_cls[0, 0, :, :]
    seg_img = seg_map[0, 0, :, :]

    res_depth = res_depth.detach().cpu().numpy()
    res_coarse = res_coarse.detach().cpu().numpy()
    res_cls = res_cls.detach().cpu().numpy()
    seg_img = seg_img.detach().cpu().numpy()

    target = lidarGt[0, 0, :, :].detach().cpu().numpy()

    # # image visualization
    # _depth_img = colored_depthmap(res_depth, 0, 255)
    # _coarse_img = colored_depthmap(res_coarse, 0, 255)
    # _cls_img = colored_depthmap(res_cls, 0, 255)
    # _seg_img = colored_depthmap(seg_img, 0, 10)
    # _depth = np.vstack([img, _coarse_img, _cls_img, _depth_img, _seg_img])
    # Image.fromarray(np.uint8(_depth)).show()
    # raise NotImplementedError

    # compute errors
    pred = np.uint8(res_depth)
    target = np.uint8(target)
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
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
    trainbar.update(1)
    # print('\n======================')
    # print('abs_rel: ', abs_rel)
    # print('sq_rel: ', sq_rel)
    # print('rmse: ', rmse)
    # print('rmse_log: ', rmse_log)
    # print('d1: ', d1)
    # print('d2: ', d2)
    # print('d3: ', d3)
    # print('mae: ', mae)
    # break
print('\n======================')
print('data_length: ', len(Data))
print('abs_rel: ', _abs_rel / len(Data))
print('sq_rel: ', _sq_rel / len(Data))
print('rmse: ', _rmse / len(Data))
print('d1: ', _d1 / len(Data))
print('d2: ', _d2 / len(Data))
print('d3: ', _d3 / len(Data))
print('mae: ', _mae / len(Data))
