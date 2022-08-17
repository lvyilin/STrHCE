import argparse
import json
import random
import time
import torch.backends.cudnn as cudnn
import torch.cuda.amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import warnings
from pathlib import Path
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter

import models
from datasets import *
from datasets.transforms_factory import create_transform
from losses import HingeLoss
from utils import *

parser = argparse.ArgumentParser(description='STrHCE Training')
parser.add_argument('data', metavar='DIR',
                    help='path to datasets')
parser.add_argument('-a', '--arch', metavar='ARCH',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--new-size', type=int, default=512)
parser.add_argument('--crop-size', type=int, default=448)
parser.add_argument('--datadir', type=str, default='.')
parser.add_argument('--logdir', type=str, default='.')
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--lr-step', type=int, default=None)
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--clip_grad', type=float, default=None)
parser.add_argument('--milestones', nargs='+', type=int, default=None)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--use-amp', action='store_true')
parser.add_argument('--lr-policy', type=str)
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--lam1', type=float, default=0.5)
parser.add_argument('--lam2', type=float, default=0.8)
parser.add_argument('--overlap', type=int, default=8)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--concept-level', type=int, choices=[2, 3])

best_acc1 = 0


def main():
    args = parser.parse_args()
    free_gpus = get_free_gpu(num=args.ngpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
    os.environ["OMP_NUM_THREADS"] = str(args.ngpus * 4)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    filepath = Path(__file__)
    copyfile(filepath.absolute(), os.path.join(args.logdir, filepath.name))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes = get_dataset_class_number(args.data)
    model_fn = getattr(models, args.arch)
    config = {'num_classes': num_classes, 'pretrained': args.pretrained,
              'img_size': args.crop_size, 'overlap': args.overlap,
              'num_parent_classes': get_dataset_parent_class_number(args.data)}
    if args.concept_level == 3:
        config['num_pparent_classes'] = get_dataset_pparent_class_number(args.data)
    model = model_fn(**config)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    criterion = [nn.CrossEntropyLoss(),
                 HingeLoss(epsilon=args.epsilon)]
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if not args.retrain:
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                # optimizer.load_state_dict(checkpoint['optimizer'])
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = torch.tensor(best_acc1).to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    new_size, crop_size = args.new_size, args.crop_size
    train_transform = create_transform(
        input_size=crop_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    train_dataset = get_dataset(args.data, root=args.datadir, train=True, transform=train_transform)

    val_dataset = get_dataset(args.data, root=args.datadir, train=False, transform=transforms.Compose([
        transforms.Resize((new_size, new_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    args.is_master = not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    if args.warmup_epochs > 0:
        print('=> pre-training')
        warmup_optimizer = torch.optim.Adam([model.module.pos_embed_new] +
                                            [model.module.pos_embed] +
                                            [model.module.cls_token] +
                                            [model.module.dist_token] +
                                            list(model.module.head.parameters()), 1e-4,
                                            weight_decay=args.weight_decay)

        for epoch in range(args.warmup_epochs):
            train(train_loader, model, criterion, warmup_optimizer, scaler, args, epoch, None)

        print('=> finish pre-training')
    set_parameter_requires_grad(model, True)

    if args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif args.lr_policy == 'milestones':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    else:
        raise ValueError('Unknown LR scheduler')

    writer = SummaryWriter(args.logdir) if args.is_master else None
    old_best_path = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, args, epoch, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, writer)

        scheduler.step()
        if writer is not None:
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.is_master:
            path = save_checkpoint_and_remove_old({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, logdir=args.logdir, old_best_path=old_best_path)
            if path is not None:
                old_best_path = path
            writer.flush()


def train(train_loader, model, criterion, optimizer, scaler, args, epoch, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    ce2_losses = AverageMeter('CE2Loss', ':.4e')
    hinge_losses = AverageMeter('HingeLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_2 = AverageMeter('Acc2@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, ce_losses, ce2_losses, hinge_losses, top1,
         top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for cri in criterion:
        cri.train()
    has_level3 = args.concept_level == 3
    ce_loss_fn, hinge_loss_fn = criterion
    end = time.time()
    for i, data in enumerate(train_loader):
        if not has_level3:
            (images, target, par_target) = data
        else:
            (images, target, par_target, ppar_target) = data
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
            par_target = par_target.cuda(args.gpu, non_blocking=True)
            if has_level3:
                ppar_target = ppar_target.cuda(args.gpu, non_blocking=True)

        # compute output 1
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = model(images)
            if not has_level3:
                x_cls, x_cls_hier, x2_cls, x2_cls_hier, att_gt = outputs
                loss1 = ce_loss_fn(x_cls, target)
                loss2 = ce_loss_fn(x_cls_hier, par_target)
                loss3 = ce_loss_fn(x2_cls, target)
                loss4 = ce_loss_fn(x2_cls_hier, par_target)
                loss5 = hinge_loss_fn(x_cls, x2_cls, target)
                loss = (1 - args.lam1) * (loss1 + args.lam2 * loss2) + \
                       args.lam1 * (loss3 + args.lam2 * loss4) + \
                       loss5
            else:
                x_cls, x_cls_hier, x_cls_hier2, x2_cls, x2_cls_hier, x2_cls_hier2, att_gt = outputs
                loss1 = ce_loss_fn(x_cls, target)
                loss2 = ce_loss_fn(x_cls_hier, par_target)
                loss3 = ce_loss_fn(x2_cls, target)
                loss4 = ce_loss_fn(x2_cls_hier, par_target)
                loss5 = hinge_loss_fn(x_cls, x2_cls, target)
                loss6 = ce_loss_fn(x_cls_hier2, ppar_target)
                loss7 = ce_loss_fn(x2_cls_hier2, ppar_target)

                loss = (1 - args.lam1) * (loss1 + args.lam2 * loss2 + args.lam2 * args.lam2 * loss6) + \
                       args.lam1 * (loss3 + args.lam2 * loss4 + args.lam2 * args.lam2 * loss7) + \
                       loss5

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(x_cls, target, topk=(1, 5))
        acc1_2, = accuracy(x2_cls, target, topk=(1,))
        if args.distributed:
            loss = reduce_tensor(loss.data, args.world_size)
            loss1 = reduce_tensor(loss1.data, args.world_size)
            loss3 = reduce_tensor(loss3.data, args.world_size)
            loss5 = reduce_tensor(loss5.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc1_2 = reduce_tensor(acc1_2, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)

        losses.update(loss.item(), images.size(0))
        ce_losses.update(loss1.item(), images.size(0))
        ce2_losses.update(loss3.item(), images.size(0))
        hinge_losses.update(loss5.item(), images.size(0))

        top1.update(acc1[0].item(), images.size(0))
        top1_2.update(acc1_2[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.is_master and i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train/1', ce_losses.avg, epoch)
        writer.add_scalar('CELosses/train/2', ce2_losses.avg, epoch)
        writer.add_scalar('HingeLosses/train', hinge_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train/1', top1.avg, epoch)
        writer.add_scalar('Accuracy1/train/2', top1_2.avg, epoch)
        writer.add_scalar('Accuracy5/train', top5.avg, epoch)


def validate(val_loader, model, criterion, args, epoch=None, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_1 = AverageMeter('Acc1@1', ':6.2f')
    top1_2 = AverageMeter('Acc2@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top1_1, top1_2, top5],
        prefix='Test: ')
    ce_loss_func = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()
    for cri in criterion:
        cri.eval()
    has_level3 = args.concept_level == 3
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if not has_level3:
                (images, target, par_target) = data
            else:
                (images, target, par_target, ppar_target) = data
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs = model(images)
            if not has_level3:
                x_cls, x_cls_hier, x2_cls, x2_cls_hier, att_gt = outputs
            else:
                x_cls, x_cls_hier, x_cls_hier2, x2_cls, x2_cls_hier, x2_cls_hier2, att_gt = outputs

            loss = ce_loss_func(x_cls, target)

            if args.distributed:
                x2_cls = reduce_tensor(x2_cls, args.world_size)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(x_cls + x2_cls, target, topk=(1, 5))
            acc1_1, = accuracy(x_cls, target, topk=(1,))
            acc1_2, = accuracy(x2_cls, target, topk=(1,))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top1_1.update(acc1_1[0].item(), images.size(0))
            top1_2.update(acc1_2[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            if args.is_master and i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc1@1 {top1_1.avg:.3f} Acc2@1 {top1_2.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top1_1=top1_1, top1_2=top1_2, top5=top5))

        if writer is not None:
            writer.add_scalar('Time/test', batch_time.avg, epoch)
            writer.add_scalar('Losses/test', losses.avg, epoch)
            writer.add_scalar('Accuracy1/test/all', top1.avg, epoch)
            writer.add_scalar('Accuracy1/test/1', top1_1.avg, epoch)
            writer.add_scalar('Accuracy1/test/2', top1_2.avg, epoch)
            writer.add_scalar('Accuracy5/test', top5.avg, epoch)
        return max([top1.avg, top1_1.avg, top1_2.avg])


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


if __name__ == '__main__':
    main()
