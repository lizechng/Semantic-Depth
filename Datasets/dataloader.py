"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read, velo_read, seg_read


def get_loader(args, dataset):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args.crop_h, args.crop_w)
    perform_transformation = not args.no_aug

    train_dataset = Dataset_loader(
            args.data_path, dataset.train_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=perform_transformation, num_samples=args.num_samples)
    val_dataset = Dataset_loader(
            args.data_path, dataset.val_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)
    val_select_dataset = Dataset_loader(
            args.data_path, dataset.selected_paths, args.input_type,
            resize=None, rotate=args.rotate, crop=crop_size,
            flip=args.flip, rescale=args.rescale, max_depth=args.max_depth,
            sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)

    train_sampler = None
    val_sampler = None
    if args.subset is not None:
        random.seed(1)
        train_idx = [i for i in random.sample(range(len(train_dataset)-1), args.subset)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        random.seed(1)
        val_idx = [i for i in random.sample(range(len(val_dataset)-1), round(args.subset*0.5))]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=int(args.val_batch_size),  sampler=val_sampler,
        shuffle=val_sampler is None, num_workers=args.nworkers_val,
        pin_memory=True, drop_last=True)
    val_selection_loader = DataLoader(
        val_select_dataset, batch_size=int(args.val_batch_size), shuffle=False,
        num_workers=args.nworkers_val, pin_memory=True, drop_last=True)
    return train_loader, val_loader, val_selection_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, data_path, dataset_type, input_type, resize,
                 rotate, crop, flip, rescale, max_depth, sparse_val=0.0, 
                 normal=False, disp=False, train=False, num_samples=None):

        # Constants
        self.use_rgb = input_type == 'rgb'
        self.datapath = data_path
        self.dataset_type = dataset_type
        self.train = train
        self.resize = resize
        self.flip = flip
        self.crop = crop
        self.rotate = rotate
        self.rescale = rescale
        self.max_depth = max_depth
        self.sparse_val = sparse_val

        # Transformations
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)

        # Names
        self.img_name = 'img'
        self.depth_name = 'depth_in'
        self.velo_name = 'velo_in'
        self.lidarGt_name = 'lidar_gt'
        self.segGt_name = 'seg_gt'

        # Define random sampler
        self.num_samples = num_samples


    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['depth_in'])


    def define_transforms(self, input_depth, input_velo, lidarGt, segGt, img=None):
        # Define random variabels
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'

        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(input_depth, output_size=self.crop)
            input_depth = F.crop(input_depth, i, j, h, w)
            input_velo = F.crop(input_velo, i, j, h, w)
            lidarGt = F.crop(lidarGt, i, j, h, w)
            segGt = F.crop(segGt, i, j, h, w)
            if hflip_input:
                input_depth, input_velo, lidarGt, segGt = F.hflip(input_depth), F.hflip(input_velo), F.hflip(lidarGt), F.hflip(segGt)

            if self.use_rgb:
                img = F.crop(img, i, j, h, w)
                if hflip_input:
                    img = F.hflip(img)
            input_depth, input_velo, lidarGt, segGt = depth_read(input_depth, self.sparse_val), \
                                                      velo_read(input_velo, self.sparse_val, True), \
                                                      depth_read(lidarGt, self.sparse_val), \
                                                      seg_read(segGt)

        else:
            input_depth, input_velo, lidarGt, segGt = self.center_crop(input_depth), self.center_crop(input_velo),\
                                                      self.center_crop(lidarGt), self.center_crop(segGt)
            if self.use_rgb:
                img = self.center_crop(img)
            input_depth, input_velo, lidarGt, segGt = depth_read(input_depth, self.sparse_val), \
                                                      velo_read(input_velo, self.sparse_val, True), \
                                                      depth_read(lidarGt, self.sparse_val), \
                                                      seg_read(segGt)
            

        return input_depth, input_velo, lidarGt, segGt, img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        sparse_depth_name = os.path.join(self.dataset_type[self.depth_name][idx])
        sparse_velo_name = os.path.join(self.dataset_type[self.velo_name][idx])
        lidarGt_name = os.path.join(self.dataset_type[self.lidarGt_name][idx])
        segGt_name = os.path.join(self.dataset_type[self.segGt_name][idx])
        with open(sparse_depth_name, 'rb') as f:
            sparse_depth = Image.open(f)
            w, h = sparse_depth.size
            # crop[0] should be h
            sparse_depth = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w)
        with open(sparse_velo_name, 'rb') as f:
            sparse_velo = Image.open(f)
            w, h = sparse_velo.size
            sparse_velo = F.crop(sparse_velo, h-self.crop[0], 0, self.crop[0], w)
        with open(lidarGt_name, 'rb') as f:
            lidarGt = Image.open(f)
            lidarGt = F.crop(lidarGt, h-self.crop[0], 0, self.crop[0], w)
        with open(segGt_name, 'rb') as f:
            segGt = Image.open(f)
            segGt = F.crop(segGt, h-self.crop[0], 0, self.crop[0], w)
        img = None
        if self.use_rgb:
            img_name = self.dataset_type[self.img_name][idx]
            with open(img_name, 'rb') as f:
                img = (Image.open(f).convert('RGB'))
            img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)

        sparse_depth_np, sparse_velo_np, lidarGt_np, segGt_np, img_pil = self.define_transforms(sparse_depth, sparse_velo, lidarGt, segGt, img)
        input_depth = self.totensor(sparse_depth_np).float()
        input_velo = self.totensor(sparse_velo_np).float()
        lidarGt = self.totensor(lidarGt_np).float()
        segGt = self.totensor(segGt_np).float()

        # input = input_depth

        if self.use_rgb:
            img_tensor = self.totensor(img_pil).float()
            # in transform.ToTensor(), img.div(255)
            img_tensor = img_tensor # *255.0 # TODO:
            input = torch.cat((input_depth, input_velo, img_tensor), dim=0)
        return input, lidarGt, segGt

