# Dataset
import Datasets
from Datasets.dataloader import Dataset_loader
import numpy as np
from PIL import Image
from Models.sgd import amend_module
# import cv2
from Models.sgd import semantic_depth_net
import json
from torch.utils.data import DataLoader
import os
import torch
from PIL import Image
import mayavi.mlab
import matplotlib.pyplot as plt
from scipy.linalg import pinv


def ptsview(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2)
    vals = 'height'
    if vals == 'height':
        col = z
    else:
        col = d
    # f = mayavi.mlab.gcf()
    # camera = f.scene.camera
    # camera.yaw(90)
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    # camera = fig.scene.camera
    # camera.yaw(90)
    # cam, foc = mayavi.mlab.move()
    # print(cam, foc)
    mayavi.mlab.points3d(x, y, z,
                         col,
                         mode='point',
                         colormap='spectral',
                         figure=fig)
    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    axes = np.array(
        [[20, 0, 0, ], [0, 20, 0], [0, 0, 20]]
    )
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.show()


def generate_colorpc(img, pc, pcimg, debug=False):
    """
    Generate the PointCloud with color
    Parameters:
        img: image
        pc: PointCloud
        pcimg: PointCloud project to image
    Return:
        pc_color: PointCloud with color e.g. X Y Z R G B
    """
    x = np.reshape(pcimg[:, 0], (-1, 1))
    y = np.reshape(pcimg[:, 1], (-1, 1))
    xy = np.hstack([x, y])

    pc_color = []
    for idx, i in enumerate(xy):
        if (i[0] > 1 and i[0] < img.shape[1]) and (i[1] > 1 and i[1] < img.shape[0]):
            bgr = img[int(i[1]), int(i[0])]
            p_color = [pc[idx][0], pc[idx][1], pc[idx][2], bgr[2], bgr[1], bgr[0]]
            pc_color.append(p_color)
    pc_color = np.array(pc_color)

    return pc_color


def save_pcd(filename, pc_color):
    """
    Save the PointCloud with color in the term of .pcd
    Parameter:
        filename: filename of the pcd file
        pc_color: PointCloud with color
    """
    f = open(filename, "w")

    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z rgb\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F F\n")
    f.write("COUNT 1 1 1 1\n")
    f.write("WIDTH {}\n".format(pc_color.shape[0]))
    f.write("HEIGHT 1\n")
    f.write("POINTS {}\n".format(pc_color.shape[0]))
    f.write("DATA ascii\n")

    for i in pc_color:
        # rgb = (int(i[3])<<16) | (int(i[4])<<8) | (int(i[5]) | 1<<24)
        # f.write("{:.6f} {:.6f} {:.6f} {}\n".format(i[0],i[1],i[2],rgb))
        f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(i[0], i[1], i[2], i[3], i[4], i[5]))

    f.close()


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dataset = Datasets.define_dataset('zhoushan', './DepthTrainData_352x1000', 'rgb')
dataset.prepare_dataset()
print(dataset.val_paths['img'][0])
train_loader = Dataset_loader(None, dataset.val_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)

model = semantic_depth_net().cuda()
best_file_name = f'./img_seg_edge/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
                 f'checkpoint_model_epoch_149.pth.tar'

# best_file_name = f'./monod/monod_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
#                  f'checkpoint_model_epoch_149.pth.tar'

print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
Data = DataLoader(train_loader, batch_size=1)
imId = 0
for input, lidarGt, segGt in Data:

    img = np.asarray(input[0, 2:, :, :])
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(img * 255))
    img = np.asarray(img)
    # ----------------------------------------------------------

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    depth_color = np.zeros((352, 1000, 3), np.uint8)
    input = input.cuda()
    res_coarse, res_cls, res_depth, seg, radar_points = model(input)
    res_depth = res_depth[0, 0, :, :] * 256
    res_gt = lidarGt[0, 0, :, :] * 256

    # depth prediction
    res_depth = res_depth.detach().cpu().numpy()
    res_gt = res_gt.detach().cpu().numpy()
    res_depth = np.uint8(res_depth)
    res_gt = np.uint8(res_gt)

    # for i in range(res_gt.shape[0]):
    #     for j in range(res_gt.shape[1]):
    #         if res_gt[i, j] > 0:
    #             color = cmap[int(res_gt[i, j]), :]
    #             cv2.circle(img,
    #                        (j, i),
    #                        1, color=tuple(color), thickness=-1)

    matrix = json.load(open('1635683808.490.json'))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']

    matrix = np.asarray(matrix)
    inv = pinv(matrix)
    pc = []
    c = []
    pc_color = []

    for i in range(res_depth.shape[0]):
        for j in range(res_depth.shape[1]):
            # if res_gt[i, j] > 0:
            v = (i * 2 + 500) * res_depth[i, j]
            u = (j * 2 + 750) * res_depth[i, j]
            p = np.dot(inv, np.asarray([u, v, res_depth[i, j]]).reshape(3, 1))
            pc.append(list(p.reshape(4, )))
            c.append(list(img[i, j]))
            pc_color.append(
                [p.reshape(4, )[0], p.reshape(4, )[1], p.reshape(4, )[2], img[i, j, 0], img[i, j, 1], img[i, j, 2]])

    pc = np.asarray(pc)
    c = np.asarray(c)
    pc_color = np.asarray(pc_color)
    save_pcd('result_img/pred.pcd', pc_color)
    # ptsview(pc)

    # ## camera coordinate
    # t = np.asarray([[ 0.0413620011242213, -0.999018744528412,   0.0158345490581776, -0.265290406655631],
    #                 [-0.0188262903381298, -0.0166245599949506, -0.999684547643445,  -0.391122231870622],
    #                 [-0.0188262903381298, -0.0166245599949506, -0.999684547643445,  -0.391122231870622],
    #                 [ 0.998966844122018,   0.0410508475655525, -0.0194954420069267, -0.515288952567310]])
    # p_cam = np.dot(pc, t)
    # print(f'p_cam: {p_cam.shape}')
    # print('Depth:', np.min(p_cam[:, 0]), np.max(p_cam[:, 0]))  # depth
    # print('LR:', np.min(p_cam[:, 1]), np.max(p_cam[:, 1]))  # left-right
    # print('Height:', np.min(p_cam[:, 2]), np.max(p_cam[:, 2]))  # height
    # img_fv = np.ones((800, 2000, 3))
    # for pi, ci in zip(pc, c):
    #     img_fv[int(-pi[2] * 10) + 600, int(-pi[1] * 5) + 100] = [ci[0], ci[1], ci[2]]
    # Image.fromarray(np.uint8(img_fv)).convert('RGB').show()

    print(f'point cloud: {pc.shape}')
    print('Depth:', np.min(pc[:, 0]), np.max(pc[:, 0]))  # depth
    print('LR:', np.min(pc[:, 1]), np.max(pc[:, 1]))  # left-right
    print('Height:', np.min(pc[:, 2]), np.max(pc[:, 2]))  # height
    img_fv = np.ones((800, 1200, 3))
    for pi, ci in zip(pc, c):
        img_fv[int(-pi[2] * 10) + 400, int(-pi[1] * 10) + 600] = [ci[0], ci[1], ci[2]]
    Image.fromarray(np.uint8(img_fv)).convert('RGB').show()
    Image.fromarray(np.uint8(img)).convert('RGB').show()
    print(imId)
    imId += 1

    # projection
    pts_2d = np.dot(pc, np.transpose(matrix))
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < 2750) & (pts_2d[:, 1] < 1204) & \
           (pts_2d[:, 0] > 750) & (pts_2d[:, 1] > 500) & \
           (pts_2d[:, 2] > 5) & (pts_2d[:, 2] <= 255)
    pts_2d = pts_2d[mask, :]
    c = c[mask, :]
    pts_2d[:, 0] -= 750
    pts_2d[:, 1] -= 500
    res = np.zeros((704, 2000, 3))
    print(c)
    print(f'c.shape: {pc.shape, pts_2d.shape, c.shape}')

    for pi, ci in zip(pts_2d, c):
        res[int(pi[1]), int(pi[0])] = [ci[0], ci[1], ci[2]]
    Image.fromarray(np.uint8(res)).convert('RGB').show()
    # Image.fromarray(image_img).show()
    break
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
