import os
from PIL import Image
import numpy as np

data_path = './DepthTestData'

if not os.path.exists('./DepthTestData_352x1000/depth/val'):
    os.makedirs('./DepthTestData_352x1000/depth/val', exist_ok=True)

if not os.path.exists('./DepthTestData_352x1000/image/val'):
    os.makedirs('./DepthTestData_352x1000/image/val', exist_ok=True)

if not os.path.exists('./DepthTestData_352x1000/velo/val'):
    os.makedirs('./DepthTestData_352x1000/velo/val', exist_ok=True)

if not os.path.exists('./DepthTestData_352x1000/lidar/val'):
    os.makedirs('./DepthTestData_352x1000/lidar/val', exist_ok=True)

if not os.path.exists('./DepthTestData_352x1000/segmentation/val'):
    os.makedirs('./DepthTestData_352x1000/segmentation/val', exist_ok=True)

typesets = os.listdir(data_path)
typesets = sorted(typesets)
for typeset in typesets:
    files = os.listdir(os.path.join(data_path, typeset, 'val'))
    files = sorted(files)
    for file in files:
        img = Image.open(os.path.join(data_path, typeset, 'val', file))
        img = np.asarray(img)
        img = img[::2, ::2]
        Image.fromarray(img).save(f'./DepthTestData_352x1000/{typeset}/val/{file}')
