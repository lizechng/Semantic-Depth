import matplotlib.pyplot as plt
import re
import numpy as np

log_path = './imgEdge/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/log_train_start_0.txt'
log_lines = open(log_path, 'r').readlines()
coarse_list, cls_list, depth_list, seg_list, edge_list = [], [], [], [], []
for log_line in log_lines:
    line = log_line.strip()
    if re.search('Epoch', line):
        coarse, cls, depth, seg, edge = list(map(float, line.split('\t')[3].split(' ')))
        coarse_list.append(coarse)
        cls_list.append(cls)
        depth_list.append(depth)
        seg_list.append(seg)
        edge_list.append(edge)

x = np.arange(0, len(cls_list))
plt.plot(x, coarse_list, x, cls_list, x, depth_list, x, seg_list, x, edge_list)
plt.legend(['coarse', 'cls', 'depth', 'seg', 'edge'])
plt.show()
