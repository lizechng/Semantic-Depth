# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt
import json

# Image processing
from PIL import Image
from PIL import ImageDraw

# cityscapes imports
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'Sky'                  ,  0 ,        0 , 'void'            , 0       , False        , True         , (190,153,153) ),
    Label(  'Buiding'             ,   1 ,        1 , 'void'            , 0       , False        , True         , (250, 10, 10) ),
    Label(  'Traffic' ,               2 ,        2 , 'void'            , 0       , False        , True         , ( 70, 70, 70) ),
    Label(  'Traffic_h'            ,  3 ,        3 , 'void'            , 0       , False        , True         , (250,170, 30) ),
    Label(  'Cycle'                ,  4 ,        4 , 'void'            , 0       , False        , True         , (  0, 20, 60) ),
    Label(  'Car'                  ,  5 ,        5 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'Road'                 ,  6 ,        6 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'BG'                   ,  7 ,        7 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'Tree'                 ,  8 ,        8 , 'flat'            , 1       , False        , False        , (244, 35,232) ),

]

name2label = {label.name: label for label in labels}


def main():
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    json_path = './Zhoushan/image/train/20211027_1_group0028/20211027_1_group0028_frame0005.json'
    img_path = './Zhoushan/image/train/20211027_1_group0028/20211027_1_group0028_frame0005.png'
    labels = json.load(open(json_path))
    labelImg = Image.new('RGB', (2000, 704), (170, 22, 40))
    img = Image.open(img_path)
    drawer = ImageDraw.Draw(labelImg)
    for itm in labels['shapes']:
        label = itm['label']
        val = name2label[label].trainId
        color = cmap[int((val+3) * 20), :]
        # print(int(color))
        points = [tuple(p) for p in itm['points']]
        drawer.polygon(points, fill=tuple(list(map(int, color))))
    img.show()
    labelImg.show()
    # labelImg.convert('L').save('./Zhoushan_mini/segmentation/train/20211027_1_group0028/20211027_1_group0028_frame0005.png')
    # import numpy as np
    # labelImg = np.asarray(labelImg)
    # img = np.asarray(img)
    # im = img*0.6 + labelImg*0.4
    # Image.fromarray(np.uint8(im)).show()


import numpy as np
def img_downsample():
    mini_path = './Zhoushan_mini'
    for cls in os.listdir(mini_path):
        train = os.path.join(mini_path, cls, 'train')
        for gp in os.listdir(train):
            for file in os.listdir(os.path.join(train, gp)):
                if 'png' not in file:
                    continue
                img = Image.open(os.path.join(train, gp, file))
                img = np.asarray(img)[::2, ::2]
                Image.fromarray(img).save(os.path.join(train, gp, file))
                # print(train, img.shape)

main()