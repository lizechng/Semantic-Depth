# Dataset
import Datasets
from Datasets.dataloader import Dataset_loader
import numpy as np
from PIL import Image
from Models.sgd import amend_module
from Models.monod import mono_depth_net
import time
from torch.utils.data import DataLoader
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dataset = Datasets.define_dataset('zhoushan', './DepthTestData_352x1000', 'rgb')
dataset.prepare_dataset()
print(dataset.val_paths['img'][0])
train_loader = Dataset_loader(None, dataset.val_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)
# Image.open(dataset.train_paths['img'][0]).show()
model = mono_depth_net().cuda()
best_file_name = f'./monod/monod_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
                 f'checkpoint_model_epoch_149.pth.tar'

# best_file_name = f'./Saved/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wlid1_wrgb1_wguide1_wpred1_patience7_num_samplesNone_multiFalse/' \
#                  f'checkpoint_model_epoch_141.pth.tar'

print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
Data = DataLoader(train_loader, batch_size=1)
for input, lidarGt, segGt in Data:
    print(f'input: {input.shape}')
    print(f'segGt: {segGt.shape}')
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
    res_depth = model(input)
    res = res_depth
    end = time.time()
    print(f'res: {res.shape}, {end - start}')

    depth = res[0, 0, :, :] * 256
    cv2.imshow('img', img)
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
