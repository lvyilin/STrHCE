import numbers
import os
import pandas as pd
import shutil
import subprocess
import torch
from io import StringIO


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_free_gpu(num=1):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df = gpu_df.sort_values(by='memory.free', ascending=False)
    print('GPU usage:\n{}'.format(gpu_df))
    if len(gpu_df) < num:
        raise RuntimeError('No enough GPU')
    free_gpus = []
    for i in range(num):
        print('Returning GPU{} with {} free MiB'.format(gpu_df.index[i], gpu_df.iloc[i]['memory.free']))
        free_gpus.append(gpu_df.index[i])
    return ','.join(str(x) for x in free_gpus)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, best_filename='checkpoint.pth', logdir=None):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), best_filename)
    torch.save(state, path)
    if is_best:
        best_filename = 'model_best_ep{:03d}_acc{:.2f}.pth'.format(state['epoch'], state['best_acc1'])
        shutil.copyfile(path, os.path.join(logdir, best_filename))


def save_checkpoint_and_remove_old(state, is_best, best_filename='checkpoint.pth', logdir=None, old_best_path=None):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), best_filename)
    torch.save(state, path)
    if is_best:
        best_filename = 'model_best_ep{:03d}_acc{:.2f}.pth'.format(state['epoch'], state['best_acc1'])
        best_path = os.path.join(logdir, best_filename)
        shutil.copyfile(path, best_path)
        if old_best_path is not None:
            os.remove(old_best_path)
        return best_path
    return None
