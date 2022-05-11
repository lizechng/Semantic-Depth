import os
from PIL import Image
import numpy as np

data_path = './DepthTrainData_704x2000'

if not os.path.exists('./DepthTrainData_352x1000/depth/train'):
    os.makedirs('./DepthTrainData_352x1000/depth/train', exist_ok=True)

if not os.path.exists('./DepthTrainData_352x1000/image/train'):
    os.makedirs('./DepthTrainData_352x1000/image/train', exist_ok=True)

if not os.path.exists('./DepthTrainData_352x1000/velo/train'):
    os.makedirs('./DepthTrainData_352x1000/velo/train', exist_ok=True)

if not os.path.exists('./DepthTrainData_352x1000/lidar/train'):
    os.makedirs('./DepthTrainData_352x1000/lidar/train', exist_ok=True)

if not os.path.exists('./DepthTrainData_352x1000/segmentation/train'):
    os.makedirs('./DepthTrainData_352x1000/segmentation/train', exist_ok=True)

typesets = os.listdir(data_path)
typesets = sorted(typesets)
for typeset in typesets:
    files = os.listdir(os.path.join(data_path, typeset, 'train'))
    files = sorted(files)
    for file in files:
        img = Image.open(os.path.join(data_path, typeset, 'train', file))
        img = np.asarray(img)
        img = img[::2, ::2]
        Image.fromarray(img).save(f'./DepthTrainData_352x1000/{typeset}/train/{file}')
