# Dataset
import Datasets
from Datasets.dataloader import Dataset_loader
import numpy as np
from PIL import Image
from Models.sgd import amend_module
from Models.sgd import semantic_depth_net
import time
from torch.utils.data import DataLoader
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def colored_depthmap(depth, d_min=None, d_max=None):
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]

if not os.path.exists('result_255/train/color'):
    os.makedirs('result_255/train/color')
if not os.path.exists('result_255/train/gray/depth'):
    os.makedirs('result_255/train/gray/depth')
if not os.path.exists('result_255/train/gray/coarse'):
    os.makedirs('result_255/train/gray/coarse')
if not os.path.exists('result_255/train/gray/cls'):
    os.makedirs('result_255/train/gray/cls')
if not os.path.exists('result_255/test/color'):
    os.makedirs('result_255/test/color')
if not os.path.exists('result_255/test/gray/depth'):
    os.makedirs('result_255/test/gray/depth')
if not os.path.exists('result_255/test/gray/coarse'):
    os.makedirs('result_255/test/gray/coarse')
if not os.path.exists('result_255/test/gray/cls'):
    os.makedirs('result_255/test/gray/cls')

model = semantic_depth_net().cuda()
# best_file_name = f'./img_seg_edge/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
#                  f'checkpoint_model_epoch_149.pth.tar'
best_file_name = f'./ImgSegEdge_255/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge1_patience7_num_samplesNone_multiFalse/' \
                 f'checkpoint_model_epoch_149.pth.tar'
print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))

dataset = Datasets.define_dataset('zhoushan', './DepthTrainData_352x1000', 'rgb')
dataset.prepare_dataset()
train_loader = Dataset_loader(None, dataset.train_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)

# Data = DataLoader(train_loader, batch_size=1)
# for i, (input, lidarGt, segGt) in enumerate(Data):
#     print(i)
#     img = np.asarray(input[0, 2:, :, :])
#     img = img.transpose(1, 2, 0)
#     img = Image.fromarray(np.uint8(img * 255))
#     img = np.asarray(img)
#     input = input.cuda()
#
#     res_coarse, res_cls, res_depth, seg, radar_points = model(input)
#
#     seg_map = torch.argmax(seg, 1)[:, None]
#
#     res_depth = res_depth[0, 0, :, :] #* 256
#     res_coarse = res_coarse[0, 0, :, :] #* 256
#     res_cls = res_cls[0, 0, :, :] #* 256
#     seg_img = seg_map[0, 0, :, :]
#
#     res_depth = res_depth.detach().cpu().numpy()
#     res_coarse = res_coarse.detach().cpu().numpy()
#     res_cls = res_cls.detach().cpu().numpy()
#     seg_img = seg_img.detach().cpu().numpy()
#     _depth_img = colored_depthmap(res_depth, 0, 255)
#     _coarse_img = colored_depthmap(res_coarse, 0, 255)
#     _cls_img = colored_depthmap(res_cls, 0, 255)
#     _seg_img = colored_depthmap(seg_img, 0, 10)
#     _depth = np.vstack([img, _coarse_img, _cls_img, _depth_img, _seg_img])
#     Image.fromarray(np.uint8(res_depth)).save(f'result_255/train/gray/depth/{dataset.train_paths["img"][i].split("/")[-1]}')
#     Image.fromarray(np.uint8(res_coarse)).save(f'result_255/train/gray/coarse/{dataset.train_paths["img"][i].split("/")[-1]}')
#     Image.fromarray(np.uint8(res_cls)).save(f'result_255/train/gray/cls/{dataset.train_paths["img"][i].split("/")[-1]}')
#
#     Image.fromarray(np.uint8(_depth)).save(f'result_255/train/color/{dataset.train_paths["img"][i].split("/")[-1]}')
#     # break


dataset = Datasets.define_dataset('zhoushan', '/media/personal_data/lizc/DepthCompletion/DepthTest', 'rgb')
dataset.prepare_dataset()
val_loader = Dataset_loader(None, dataset.val_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)
Data = DataLoader(val_loader, batch_size=1)
for i, (input, lidarGt, segGt) in enumerate(Data):
    print(i)
    img = np.asarray(input[0, 2:, :, :])
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(img * 255))
    img = np.asarray(img)
    input = input.cuda()

    res_coarse, res_cls, res_depth, seg, radar_points = model(input)

    seg_map = torch.argmax(seg, 1)[:, None]

    res_depth = res_depth[0, 0, :, :] #* 256
    res_coarse = res_coarse[0, 0, :, :] #* 256
    res_cls = res_cls[0, 0, :, :] #* 256
    seg_img = seg_map[0, 0, :, :]

    res_depth = res_depth.detach().cpu().numpy()
    res_coarse = res_coarse.detach().cpu().numpy()
    res_cls = res_cls.detach().cpu().numpy()
    seg_img = seg_img.detach().cpu().numpy()
    _depth_img = colored_depthmap(res_depth, 0, 255)
    _coarse_img = colored_depthmap(res_coarse, 0, 255)
    _cls_img = colored_depthmap(res_cls, 0, 255)
    _seg_img = colored_depthmap(seg_img, 0, 10)
    _depth = np.vstack([img, _coarse_img, _cls_img, _depth_img, _seg_img])
    Image.fromarray(np.uint8(res_depth)).save(f'result_255/test/gray/depth/{dataset.val_paths["img"][i].split("/")[-1]}')
    Image.fromarray(np.uint8(res_coarse)).save(f'result_255/test/gray/coarse/{dataset.val_paths["img"][i].split("/")[-1]}')
    Image.fromarray(np.uint8(res_cls)).save(f'result_255/test/gray/cls/{dataset.val_paths["img"][i].split("/")[-1]}')

    Image.fromarray(np.uint8(_depth)).save(f'result_255/test/color/{dataset.val_paths["img"][i].split("/")[-1]}')
    # break
