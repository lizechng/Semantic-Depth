"""
Author: Zecheng Li
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
# from .ERFNet import Net
from Models.ERFNet import Net

class semantic_depth_net(nn.Module):
    def __init__(self):
        super(semantic_depth_net, self).__init__()

        self.combine = 'concat'

        self.amend_radar = amend_module(2)
        self.amend_rgb = amend_module(3)
        self.amend_time = time_module(3)
        self.r2_fuse = fuse_module(2, 3)

        # coarse depth(1), segmentation map(9), radar point confidence(1), global feature(1)
        out_channels = 12
        self.backbone = Net(in_channels=4, out_channels=out_channels)
        local_channels_in = 2 if self.combine == 'concat' else 1
        self.convbnrelu = nn.Sequential(convbn(local_channels_in, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))

        self.with_seed_points = True
        if not self.with_seed_points:
            out_chan = 1
            self.depth_cls0 = cls_depth(out_chan, 32)
            self.depth_cls1 = cls_depth(out_chan, 32)
            self.depth_cls2 = cls_depth(out_chan, 32)
            self.depth_cls3 = cls_depth(out_chan, 32)
            self.depth_cls4 = cls_depth(out_chan, 32)
            self.depth_cls5 = cls_depth(out_chan, 32)
            self.depth_cls6 = cls_depth(out_chan, 32)
            self.depth_cls7 = cls_depth(out_chan, 32)
            self.depth_cls8 = cls_depth(out_chan, 32)
        else:
            out_chan = 3
            self.depth_cls0 = cls_region_gd(out_chan, 32)
            self.depth_cls1 = cls_region_gd(out_chan, 32)
            self.depth_cls2 = cls_region_gd(out_chan, 32)
            self.depth_cls3 = cls_region_gd(out_chan, 32)
            self.depth_cls4 = cls_region_gd(out_chan, 32)
            self.depth_cls5 = cls_region_gd(out_chan, 32)
            self.depth_cls6 = cls_region_gd(out_chan, 32)
            self.depth_cls7 = cls_region_gd(out_chan, 32)
            self.depth_cls8 = cls_region_gd(out_chan, 32)

        self.depth_fusion = nn.Sequential(convbn(2, 32, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input, epoch=50):
        rgb_in = input[:, 2:, :, :]
        radar_in = input[:, 0:2, :, :]

        # Part I: Radar points correction and map prediction
        sigma = torch.nn.Parameter(torch.Tensor([1.0])).to(input.device) # Time variable variance

        # 1. Tangential velocity estimation
        amend_radar = self.amend_radar(radar_in)
        amend_rgb = self.amend_rgb(rgb_in)
        r2_fuse = self.r2_fuse(amend_radar, amend_rgb)
        # 2. Sensor time calibration error
        amend_time = self.amend_time(rgb_in)
        amend_time = (amend_time / sigma).exp()
        # 3. Horizontal offset correction and radial distance correction
        amend_shift = r2_fuse * amend_time
        amend_depth = radar_in[:, 1:2, :, :] * amend_time + radar_in[:, 0:1, :, :]
        # 4. Point-by-Point correction based on prediction map
        orign = list(torch.where(amend_depth > 0))
        index = list(torch.where(amend_depth > 0))

        shift0 = amend_shift[:, 0:1, :, :][index]
        shift1 = amend_shift[:, 1:2, :, :][index]

        res = torch.zeros(amend_depth.shape, device=amend_depth.device)
        index[-1] = (index[-1] + shift0).long()
        index[-2] = (index[-2] + shift1).long()

        select = torch.where((index[-1] < amend_depth.shape[-1]) & (index[-1] > 0) &
                              (index[-2] < amend_depth.shape[-2]) & (index[-2] > 0))

        index[0] = index[0][select]
        index[1] = index[1][select]
        index[2] = index[2][select]
        index[3] = index[3][select]

        orign[0] = orign[0][select]
        orign[1] = orign[1][select]
        orign[2] = orign[2][select]
        orign[3] = orign[3][select]

        res[index] = amend_depth[orign]

        # BackBone
        input = torch.cat((res, rgb_in), 1)

        embedding0, embedding1, embedding2 = self.backbone(input)

        global_feature = embedding0[:, 0:1, :, :] # Adding dimensionality to global features
        coarse_depth = embedding0[:, 1:2, :, :]
        conf = embedding0[:, 2:3, :, :]
        segmap = embedding0[:, 3:12, :, :]

        # Part II
        # 1. Adjust radar points based on confidence map
        conf = conf.sigmoid()
        conf = torch.where(conf > 0.5, 1, 0)
        radar_points = res * conf
        if self.combine == 'concat':
            input = torch.cat((radar_points, global_feature), 1)
        else:
            raise NotImplemented
        out = self.convbnrelu(input)
        # 2. Local depth complementation based on semantic segmentation map
        # 2.1: Regional Depth complementation base on radar points.
        if self.with_seed_points:
            meshx = torch.arange(0, radar_points.shape[2]).reshape(radar_points.shape[2], -1).to(out.device)
            meshx = meshx.repeat(1, radar_points.shape[3])
            meshy = torch.arange(0, radar_points.shape[3]).reshape(-1, radar_points.shape[3]).to(out.device)
            meshy = meshy.repeat(radar_points.shape[2], 1)
            mesh = torch.cat([meshx[None], meshy[None]], 0)
            mesh = mesh.expand(res.shape[0], 2, mesh.shape[1], mesh.shape[2]) # mesh.requires_grad False

        mask_cls0 = torch.argmax(segmap, 1) == 0
        mask_cls0 = mask_cls0[:, None, :, :]
        input_cls0 = torch.where(mask_cls0, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point0 = torch.where(mask_cls0, radar_points, torch.full_like(radar_points, 0))
            depth_cls0 = self.depth_cls0(input_cls0, embedding1, embedding2, radar_point0, mesh)
        else:
            depth_cls0 = self.depth_cls0(input_cls0, embedding1, embedding2)
        depth_cls0 = torch.where(mask_cls0, depth_cls0, torch.full_like(depth_cls0, 0))

        mask_cls1 = torch.argmax(segmap, 1) == 1
        mask_cls1 = mask_cls1[:, None, :, :]
        input_cls1 = torch.where(mask_cls1, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point1 = torch.where(mask_cls1, radar_points, torch.full_like(radar_points, 0))
            depth_cls1 = self.depth_cls0(input_cls1, embedding1, embedding2, radar_point1, mesh)
        else:
            depth_cls1 = self.depth_cls0(input_cls1, embedding1, embedding2)
        depth_cls1 = torch.where(mask_cls1, depth_cls1, torch.full_like(depth_cls1, 0))

        mask_cls2 = torch.argmax(segmap, 1) == 2
        mask_cls2 = mask_cls2[:, None, :, :]
        input_cls2 = torch.where(mask_cls2, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point2 = torch.where(mask_cls2, radar_points, torch.full_like(radar_points, 0))
            depth_cls2 = self.depth_cls0(input_cls2, embedding1, embedding2, radar_point2, mesh)
        else:
            depth_cls2 = self.depth_cls0(input_cls2, embedding1, embedding2)
        depth_cls2 = torch.where(mask_cls2, depth_cls2, torch.full_like(depth_cls2, 0))

        mask_cls3 = torch.argmax(segmap, 1) == 3
        mask_cls3 = mask_cls3[:, None, :, :]
        input_cls3 = torch.where(mask_cls3, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point3 = torch.where(mask_cls3, radar_points, torch.full_like(radar_points, 0))
            depth_cls3 = self.depth_cls0(input_cls3, embedding1, embedding2, radar_point3, mesh)
        else:
            depth_cls3 = self.depth_cls0(input_cls3, embedding1, embedding2)
        depth_cls3 = torch.where(mask_cls3, depth_cls3, torch.full_like(depth_cls3, 0))

        mask_cls4 = torch.argmax(segmap, 1) == 4
        mask_cls4 = mask_cls4[:, None, :, :]
        input_cls4 = torch.where(mask_cls4, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point4 = torch.where(mask_cls4, radar_points, torch.full_like(radar_points, 0))
            depth_cls4 = self.depth_cls0(input_cls4, embedding1, embedding2, radar_point4, mesh)
        else:
            depth_cls4 = self.depth_cls0(input_cls4, embedding1, embedding2)
        depth_cls4 = torch.where(mask_cls4, depth_cls4, torch.full_like(depth_cls4, 0))

        mask_cls5 = torch.argmax(segmap, 1) == 5
        mask_cls5 = mask_cls5[:, None, :, :]
        input_cls5 = torch.where(mask_cls5, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point5 = torch.where(mask_cls5, radar_points, torch.full_like(radar_points, 0))
            depth_cls5 = self.depth_cls0(input_cls5, embedding1, embedding2, radar_point5, mesh)
        else:
            depth_cls5 = self.depth_cls0(input_cls5, embedding1, embedding2)
        depth_cls5 = torch.where(mask_cls5, depth_cls5, torch.full_like(depth_cls5, 0))

        mask_cls6 = torch.argmax(segmap, 1) == 6
        mask_cls6 = mask_cls6[:, None, :, :]
        input_cls6 = torch.where(mask_cls6, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point6 = torch.where(mask_cls6, radar_points, torch.full_like(radar_points, 0))
            depth_cls6 = self.depth_cls0(input_cls6, embedding1, embedding2, radar_point6, mesh)
        else:
            depth_cls6 = self.depth_cls0(input_cls6, embedding1, embedding2)
        depth_cls6 = torch.where(mask_cls6, depth_cls6, torch.full_like(depth_cls6, 0))

        mask_cls7 = torch.argmax(segmap, 1) == 7
        mask_cls7 = mask_cls7[:, None, :, :]
        input_cls7 = torch.where(mask_cls7, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point7 = torch.where(mask_cls7, radar_points, torch.full_like(radar_points, 0))
            depth_cls7 = self.depth_cls0(input_cls7, embedding1, embedding2, radar_point7, mesh)
        else:
            depth_cls7 = self.depth_cls0(input_cls7, embedding1, embedding2)
        depth_cls7 = torch.where(mask_cls7, depth_cls7, torch.full_like(depth_cls7, 0))

        mask_cls8 = torch.argmax(segmap, 1) == 8
        mask_cls8 = mask_cls8[:, None, :, :]
        input_cls8 = torch.where(mask_cls8, out, torch.full_like(out, -1))
        if self.with_seed_points:
            radar_point8 = torch.where(mask_cls8, radar_points, torch.full_like(radar_points, 0))
            depth_cls8 = self.depth_cls0(input_cls8, embedding1, embedding2, radar_point8, mesh)
        else:
            depth_cls8 = self.depth_cls0(input_cls8, embedding1, embedding2)
        depth_cls8 = torch.where(mask_cls8, depth_cls8, torch.full_like(depth_cls8, 0))

        depth_cls = depth_cls0 + depth_cls1 + depth_cls2 + depth_cls3 + depth_cls4 + depth_cls5 + depth_cls6 + depth_cls7 + depth_cls8

        # Part III: Depth fusion
        depth_input = torch.cat((coarse_depth, depth_cls), 1)
        depth = self.depth_fusion(depth_input)

        return coarse_depth, depth_cls, depth, segmap, radar_points

class cls_depth(nn.Module):
    def __init__(self, out_chan, in_chan):
        super(cls_depth, self).__init__()
        self.hourglass1 = hourglass_1(in_chan)
        # self.hourglass2 = hourglass_2(in_chan)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input, embedding1, embedding2):
        out1, embedding3, embedding4 = self.hourglass1(input, embedding1, embedding2)
        out1 = out1 + input
        # out2 = self.hourglass2(out1, embedding3, embedding4)
        # out2 = out2 + out1
        res = self.fuse(out1)
        return res

class cls_region_gd(nn.Module):
    def __init__(self, out_chan, in_chan):
        super(cls_region_gd, self).__init__()
        self.hourglass1 = hourglass_1(in_chan)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input, embedding1, embedding2, radar_points, mesh):
        out1, embedding3, embedding4 = self.hourglass1(input, embedding1, embedding2)
        out1 = out1 + input
        res = self.fuse(out1)
        seed_select = res[:, :2, :, :]
        offset_seed = res[:, 2:, :, :]
        depth_cls = torch.zeros(radar_points.shape, device=radar_points.device)
        mesh_seed = mesh + seed_select
        mesh_seed[:, 0:1, :, :] = torch.where(mesh_seed[:, 0:1, :, :] < 0,
                                              mesh_seed[:, 0:1, :, :],
                                              torch.full_like(mesh_seed[:, 0:1, :, :], 0))
        mesh_seed[:, 0:1, :, :] = torch.where(mesh_seed[:, 0:1, :, :] > radar_points.shape[2] - 1,
                                              mesh_seed[:, 0:1, :, :],
                                              torch.full_like(mesh_seed[:, 0:1, :, :], radar_points.shape[2] - 1))
        mesh_seed[:, 1:2, :, :] = torch.where(mesh_seed[:, 1:2, :, :] < 0,
                                              mesh_seed[:, 1:2, :, :],
                                              torch.full_like(mesh_seed[:, 1:2, :, :], 0))
        mesh_seed[:, 1:2, :, :] = torch.where(mesh_seed[:, 1:2, :, :] > radar_points.shape[3] - 1,
                                              mesh_seed[:, 1:2, :, :],
                                              torch.full_like(mesh_seed[:, 1:2, :, :], radar_points.shape[3] - 1))
        # torch.clamp will make the GPU memory gradually increase with the training process
        # mesh_seed[:, 0:1, :, :] = torch.clamp(mesh_seed[:, 0:1, :, :], 0, radar_points.shape[2] - 1)
        # mesh_seed[:, 1:2, :, :] = torch.clamp(mesh_seed[:, 1:2, :, :], 0, radar_points.shape[3] - 1)
        for i in range(offset_seed.shape[0]):
            depth_cls[i, 0] = radar_points[i, 0, mesh[i, 0, :, :].long(), mesh[i, 0, :, :].long()] + offset_seed[i, 0]
        return offset_seed

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
                         # nn.BatchNorm2d(out_planes))

class amend_module(nn.Module):
    def __init__(self, channels_in):
        super(amend_module, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True)) # stride 2 -> 1

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True)) # stride 2 -> 1

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.Conv2d(channels_in*3, channels_in*2, kernel_size=3, padding=1, stride=1,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True)) # stride 2 -> 1

        self.conv6 = nn.Sequential(nn.Conv2d(channels_in*2, channels_in, kernel_size=3, padding=1, stride=1,bias=False),
                                   nn.BatchNorm2d(channels_in)) # stride 2 -> 1

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, input), 1)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, input), 1)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out

class time_module(nn.Module):
    def __init__(self, channels_in):
        super(time_module, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))  # stride 2 -> 1

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))  # stride 2 -> 1

        self.conv4 = nn.Sequential(convbn(channels_in * 2, channels_in * 2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(channels_in * 3, channels_in * 2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels_in * 2),
            nn.ReLU(inplace=True))  # stride 2 -> 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(channels_in * 2, 1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1))  # stride 2 -> 1

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, input), 1)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, input), 1)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out

class fuse_module(nn.Module):
    def __init__(self, radar_channel, rgb_channel):
        super(fuse_module, self).__init__()
        chn = radar_channel + rgb_channel
        self.fuse = nn.Sequential(convbn(chn, 32, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, radar_feature, rgb_feature):
        x = torch.cat((radar_feature, rgb_feature), 1)
        x = self.fuse(x)

        return x

class hourglass_1(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_1, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, em1), 1)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, em2), 1)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out, x, x_prime

class hourglass_2(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_2, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out



if __name__ == '__main__':
    batch_size = 4
    in_channels = 4
    H, W = 256, 1216
    model = semantic_depth_net().cuda()
    print(model)
    print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in model.parameters())/1e6))
    input = torch.rand((batch_size, in_channels, H, W)).cuda().float()
    out = model(input)
    print(out[0].shape)
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='UncertainNet') as w:
    #     w.add_graph(model, input)
