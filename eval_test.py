import Datasets
from Datasets.dataloader import Dataset_loader
import torch
from Utils.utils import str2bool, define_optim, define_scheduler, \
    Logger, AverageMeter, first_run, mkdir_if_missing, \
    define_init_weights, init_distributed_mode
from Loss.benchmark_metrics import Metrics, allowed_metrics
from tqdm import tqdm
from Models.sgd import semantic_depth_net
from torch.utils.data import DataLoader

def validate(loader, model, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    metric = Metrics(max_depth=255, disp=False, normal=True)
    score = AverageMeter()
    score_1 = AverageMeter()
    # Evaluate model
    model.eval()
    # Only forward pass, hence no grads needed
    with torch.no_grad():
        # end = time.time()
        for i, (input, lidarGt, segGt) in tqdm(enumerate(loader)):

            input, lidarGt, segGt = input.cuda(non_blocking=True), lidarGt.cuda(non_blocking=True), segGt.cuda(
                    non_blocking=True)
            coarse_depth, depth_cls, depth, segmap, _ = model(input, epoch)

            metric.calculate(depth[:, 0:1], lidarGt)
            score.update(metric.get_metric('rmse'), metric.num)
            score_1.update(metric.get_metric('mae'), metric.num)

        print("===> Average RMSE score on validation set is {:.4f}".format(score.avg))
        print("===> Average MAE score on validation set is {:.4f}".format(score_1.avg))
    return score.avg, score_1.avg, losses.avg


dataset = Datasets.define_dataset('zhoushan', './DepthTrainData_352x1000', 'rgb')
dataset.prepare_dataset()

valid_dataset = Dataset_loader(None, dataset.val_paths, 'rgb', None,
                              rotate=None, crop=(352, 1000), flip=None, rescale=None, max_depth=None, sparse_val=0.0,
                              normal=False, disp=False, train=False, num_samples=None)
valid_loader = DataLoader(valid_dataset, batch_size=1)
model = semantic_depth_net().cuda()
best_file_name = f'./img_seg_edge/sdn_adam_mse_0.001_rgb_batch2_pretrainTrue_wcoarse0.7_wcls0.7_wdepth1_wseg1_wedge0.1_patience7_num_samplesNone_multiFalse/' \
                 f'model_best_epoch_138.pth.tar'

print("=> loading checkpoint '{}'".format(best_file_name))
checkpoint = torch.load(best_file_name)
model.load_state_dict(checkpoint['state_dict'])
lowest_loss = checkpoint['loss']
best_epoch = checkpoint['best epoch']
print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))

score_valid, score_valid_1, losses_valid = validate(valid_loader, model)
# print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
# print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))