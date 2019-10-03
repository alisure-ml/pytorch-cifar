import argparse
import os
import random
import shutil
import time
import warnings
from alisuretool.Tools import Tools

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch import multiprocessing as mp
import torchvision.transforms as transforms


class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()
        pass

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        pass

    def __str__(self):
        s = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return s.format(**self.__dict__)

    pass


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        pass

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        Tools.print(' '.join(entries))
        pass

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    pass


best_acc1 = 0
model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__")and callable(models.__dict__[name]))


def argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-data', metavar='DIR', default="/home/z840/data/DATASET/ILSVRC2015/Data/CLS-LOC")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total batch size of all GPUs '
                             'on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='lr', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='wd', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multi-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch N processes per node. This is the '
                             'fastest way to use PyTorch for either single node or multi node data parallel training')
    return parser


def main():
    args = argument_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        pass

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
        pass

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        pass

    args.distributed = args.world_size > 1 or args.multi_distributed

    n_gpu_per_node = torch.cuda.device_count()
    if args.multi_distributed:
        # Since we have n_gpu_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = n_gpu_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=n_gpu_per_node, args=(n_gpu_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, n_gpu_per_node, args)
        pass

    pass


def main_worker(gpu, n_gpu_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        Tools.print("Use GPU: {} for training".format(args.gpu))
        pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multi_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * n_gpu_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        pass

    # create model
    if args.pretrained:
        Tools.print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        Tools.print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        pass

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always
        # set the single device scope, otherwise, DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel,
            # we need to divide the batch size ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / n_gpu_per_node)
            args.workers = int((args.workers + n_gpu_per_node - 1) / n_gpu_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set.
            model = torch.nn.parallel.DistributedDataParallel(model)
        pass
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        pass

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)  # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            Tools.print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)  # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=loc)
                pass
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)  # best_acc1 may be from a checkpoint from a different GPU
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Tools.print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            Tools.print("=> no checkpoint found at '{}'".format(args.resume))
        pass

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val_new')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            pass

        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)  # train for one epoch
        acc1 = validate(val_loader, model, criterion, args)  # evaluate on validation set

        is_best = acc1 > best_acc1  # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

        if not args.multi_distributed or (args.multi_distributed and args.rank % n_gpu_per_node == 0):
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                             'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, is_best)
            pass

        pass

    pass


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses,
                                                 top1, top5], prefix="Epoch:[{}]".format(epoch))
    model.train()

    def each_iter(images, target):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, top_k=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pass

    end = time.time()
    for i, (_images, _target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        end = time.time()

        each_iter(_images, _target)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            pass
        pass

    pass


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    model.eval()
    with torch.no_grad():

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                pass
            pass

        # TODO: this should also be done with the ProgressMeter
        Tools.print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        pass

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    pass


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    pass


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    pass


if __name__ == '__main__':
    main()
    pass
