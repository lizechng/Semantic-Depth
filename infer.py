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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dataset = Datasets.define_dataset('zhoushan', './Zhoushan_mini', 'rgb')
dataset.prepare_dataset()
print(dataset.train_paths['img'][0])
train_loader = Dataset_loader(None, dataset.train_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)
# Image.open(dataset.train_paths['img'][0]).show()
model = semantic_depth_net().cuda()
best_file_name = f'./Saved/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wlid1_wrgb1_wguide1_wpred1_patience7_num_samplesNone_multiFalse/' \
                 f'checkpoint_model_epoch_140.pth.tar'

print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
Data = DataLoader(train_loader, batch_size=2)
for input, lidarGt, segGt in Data:
    print(f'input: {input.shape}')
    # ----------------------------------------------------------
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2

    img = np.asarray(input[0, 2:, :, :])
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(img * 255))
    img = np.asarray(img)
    # ----------------------------------------------------------
    origin_pts = np.asarray(input[0, 0, :, :] * 256)
    # print('input: ', origin_pts.shape, img.shape)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    depth_color = np.zeros((352, 1000, 3), np.uint8)
    seg_color = np.zeros((352, 1000, 3), np.uint8)
    input = input.cuda()
    start = time.time()
    res, seg, radar_points = model(input)[2:5]
    end = time.time()
    print(f'res: {res.shape}, {end - start}, {seg.shape}')
    seg_map = torch.argmax(seg, 1)[:, None]

    depth = res[0, 0, :, :] * 256
    seg_img = seg_map[0, 0, :, :]
    radar_points = radar_points[0, 0, :, :] * 256
    # radar points correction
    radar_points = radar_points.detach().cpu().numpy()
    radar_points = np.uint8(radar_points)
    # img = np.uint8(img)
    # rpts_color = img
    cv2.imshow('img', img)
    ori = img.copy()
    cor = img.copy()
    # print(f'radar points: {type(img)}, {radar_points.shape}')
    for i in range(radar_points.shape[0]):
        for j in range(radar_points.shape[1]):
            if radar_points[i, j] > 5 and radar_points[i, j] < 80:
                color = cmap[int(radar_points[i, j] * 3), :]
                cv2.circle(cor,
                           (j, i),
                           2, color=tuple(color), thickness=-1)
    for i in range(radar_points.shape[0]):
        for j in range(radar_points.shape[1]):
            if origin_pts[i, j] > 5 and origin_pts[i, j] < 80:
                color_ori = cmap[int(origin_pts[i, j] * 3), :]
                cv2.circle(ori,
                           (j, i),
                           2, color=tuple(color_ori), thickness=-1)
    cv2.imshow('cor', cor)
    cv2.imshow('ori', ori)
    # depth prediction
    depth = depth.detach().cpu().numpy()
    depth = np.uint8(depth)
    depth_list = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth_list.append(np.uint8(depth[i, j]))
            color = cmap[int(depth[i, j]), :]
            cv2.circle(depth_color,
                       (j, i),
                       1, color=tuple(color), thickness=-1)
    cv2.imshow('depth', depth_color)

    # segmentation prediction
    seg_img = seg_img.detach().cpu().numpy()
    seg_img = np.uint8(seg_img)
    for i in range(seg_img.shape[0]):
        for j in range(seg_img.shape[1]):
            depth_list.append(np.uint8(seg_img[i, j]))
            color = cmap[int(seg_img[i, j] * 30), :]
            cv2.circle(seg_color,
                       (j, i),
                       1, color=tuple(color), thickness=-1)
    cv2.imshow('seg', seg_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(set(depth_list))
    # lidar = lidarGt[0, 0, :, :] * 256
    # lidar = lidar.detach().cpu().numpy()
    # Image.fromarray(np.uint8(depth)).show()
    # Image.fromarray(np.uint8(lidar)).show()

    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='amend_module') as w:
    #     w.add_graph(model, input)

    # print(input.shape, lidarGt.shape, segGt.shape)
    # img = input[2:, :, :]
    # # depth = input[0, :, :]
    # # velo = input[1, :, :]
    # print(img.shape)
    # # depth = np.asarray(depth)
    # img = np.asarray(img)
    # img = img.transpose(1, 2, 0)
    # print(img.shape, type(img))
    # # print(depth.max()*256)
    # Image.fromarray(np.uint8(img)).show()
    # break
