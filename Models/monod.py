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

class mono_depth_net(nn.Module):
    def __init__(self):
        super(mono_depth_net, self).__init__()

        # depth(1)
        out_channels = 1
        self.backbone = Net(in_channels=3, out_channels=out_channels)

    def forward(self, input, epoch=50):
        rgb_in = input[:, 2:, :, :]

        embedding0, embedding1, embedding2 = self.backbone(rgb_in)

        depth = embedding0[:, 0:1, :, :]

        return depth

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
    in_channels = 5
    H, W = 352, 1000
    model = mono_depth_net().cuda()
    print(model)
    print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in model.parameters())/1e6))
    input = torch.rand((batch_size, in_channels, H, W)).cuda().float()
    out = model(input)
    print(out[0].shape)
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='UncertainNet') as w:
    #     w.add_graph(model, input)
