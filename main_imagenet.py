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


class RunnerMP(object):

    def __init__(self, lr=0.1, print_freq=10,
                 checkpoint_filename="./checkpoint_imagenet/ResNet18/checkpoint.pth.tar",
                 best_checkpoint_filename="./checkpoint_imagenet/ResNet18/checkpoint_best.pth.tar"):
        self.checkpoint_filename = checkpoint_filename
        self.best_checkpoint_filename = best_checkpoint_filename

        self.lr = lr
        self.print_freq = print_freq

        self.best_acc1 = 0
        self.args = self.argument_parser().parse_args()
        self._init_sth()

        self.model = self._create_model()

        self.criterion = nn.CrossEntropyLoss().cuda(self.args.gpu)  # define loss function (criterion) and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr,
                                         momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        self._resume()
        self.train_loader, self.train_sampler, self.val_loader = self._data()

        pass

    @staticmethod
    def argument_parser():
        model_names = sorted(name for name in models.__dict__ if name.islower()
                             and not name.startswith("__") and callable(models.__dict__[name]))

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
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='wd',
                            dest='weight_decay')
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
        parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url for distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
        parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
        parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
        parser.add_argument('--multi-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch N processes per node. This is the '
                                 'fastest way to use PyTorch for either single node or multi node data parallel training')
        return parser

    def _init_sth(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            cudnn.deterministic = True
            pass

        if self.args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
            pass

        if self.args.dist_url == "env://" and self.args.world_size == -1:
            self.args.world_size = int(os.environ["WORLD_SIZE"])
            pass

        if self.args.gpu is not None:
            Tools.print("Use GPU: {} for training".format(self.args.gpu))
            pass

        self.args.n_gpu_per_node = torch.cuda.device_count()
        self.args.distributed = self.args.world_size > 1 or self.args.multi_distributed

        if self.args.distributed:
            if self.args.dist_url == "env://" and self.args.rank == -1:
                self.args.rank = int(os.environ["RANK"])
            if self.args.multi_distributed:
                # For multiprocessing distributed training, rank needs to be the global rank among all the processes
                self.args.rank = self.args.rank * self.args.n_gpu_per_node + self.args.gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)
            pass

        pass

    def _create_model(self):
        # create model
        if self.args.pretrained:
            Tools.print("=> using pre-trained model '{}'".format(self.args.arch))
            model = models.__dict__[self.args.arch](pretrained=True)
        else:
            Tools.print("=> creating model '{}'".format(self.args.arch))
            model = models.__dict__[self.args.arch]()
            pass

        if self.args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor should always
            # set the single device scope, otherwise, DistributedDataParallel will use all available devices.
            if self.args.gpu is not None:
                torch.cuda.set_device(self.args.gpu)
                model.cuda(self.args.gpu)
                # When using a single GPU per process and per DistributedDataParallel,
                # we need to divide the batch size ourselves based on the total number of GPUs we have
                self.args.batch_size = int(self.args.batch_size / self.args.n_gpu_per_node)
                self.args.workers = int((self.args.workers + self.args.n_gpu_per_node - 1) / self.args.n_gpu_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu])
            else:
                model.cuda()
                # DataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set.
                model = torch.nn.parallel.DistributedDataParallel(model)
            pass
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            model = model.cuda(self.args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if self.args.arch.startswith('alexnet') or self.args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
            pass

        return model

    def _resume(self):
        # optionally resume from a checkpoint
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                Tools.print("=> loading checkpoint '{}'".format(self.args.resume))
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                else:
                    loc = 'cuda:{}'.format(self.args.gpu)  # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.args.resume, map_location=loc)
                    pass
                self.args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if self.args.gpu is not None:
                    self.best_acc1 = best_acc1.to(self.args.gpu)  # best_acc1
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                Tools.print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
            else:
                Tools.print("=> no checkpoint found at '{}'".format(self.args.resume))
            pass

        cudnn.benchmark = True
        pass

    def _data(self):
        # Data loading code
        train_dir = os.path.join(self.args.data, 'train')
        val_dir = os.path.join(self.args.data, 'val_new')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if self.args.distributed else None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=sampler)

        test_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size,
                                                 shuffle=False, num_workers=self.args.workers, pin_memory=True)
        return train_loader, sampler, val_loader

    def _save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint_filename)
        if is_best:
            shutil.copyfile(self.checkpoint_filename, self.best_checkpoint_filename)
        pass

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        pass

    @staticmethod
    def _accuracy(output, target, top_k=(1,)):
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

    def main(self):
        if self.args.multi_distributed:
            # Since we have n_gpu_per_node processes per node, the total world_size needs to be adjusted accordingly
            self.args.world_size = self.args.n_gpu_per_node * self.args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
            mp.spawn(self.train, nprocs=self.args.n_gpu_per_node, args=(self.args.n_gpu_per_node,))
        else:
            # Simply call main_worker function
            self.train()
            pass
        pass

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
                pass

            self._adjust_learning_rate(epoch)

            self.train_one_epoch(epoch)

            acc1 = self.validate()
            self.best_acc1 = max(acc1, self.best_acc1)

            if not self.args.multi_distributed or (
                    self.args.multi_distributed and self.args.rank % self.args.n_gpu_per_node == 0):
                self._save_checkpoint({'epoch': epoch + 1, 'arch': self.args.arch,
                                       'state_dict': self.model.state_dict(), 'best_acc1': self.best_acc1,
                                       'optimizer': self.optimizer.state_dict()}, acc1 > self.best_acc1)
                pass

            pass
        pass

    def train_one_epoch(self, epoch):
        batch_time, data_time = AverageMeter('Time', ':6.3f'), AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.train_loader),
                                 [batch_time, data_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch))
        self.model.train()

        def _each_iter(images, target):
            images = images.cuda(self.args.gpu, non_blocking=True) if self.args.gpu is not None else images
            target = target.cuda(self.args.gpu, non_blocking=True)

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pass

        end = time.time()
        for i, (_images, _target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            end = time.time()

            _each_iter(_images, _target)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                progress.display(i)
                pass
            pass

        pass

    def validate(self):
        batch_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

        self.model.eval()
        with torch.no_grad():

            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.cuda(self.args.gpu, non_blocking=True) if self.args.gpu is not None else images
                target = target.cuda(self.args.gpu, non_blocking=True)

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.display(i)
                    pass
                pass

            Tools.print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            pass

        return top1.avg

    pass


class RunnerSingle(object):

    def __init__(self, lr=0.1, print_freq=10, start_epoch=0, epochs=15,
                 batch_size=256, workers=30, momentum=0.9, weight_decay=1e-4, arch="resnet18",
                 data_root="/home/z840/data/DATASET/ILSVRC2015/Data/CLS-LOC",
                 resume_filename="./checkpoint_imagenet/ResNet18/checkpoint.pth.tar",
                 checkpoint_filename="./checkpoint_imagenet/ResNet18/checkpoint.pth.tar",
                 best_checkpoint_filename="./checkpoint_imagenet/ResNet18/checkpoint_best.pth.tar"):
        self.resume_filename = resume_filename
        self.checkpoint_filename = checkpoint_filename
        self.best_checkpoint_filename = best_checkpoint_filename

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.arch = arch
        self.has_pretrained = False
        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.best_acc1 = 0

        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss().cuda()  # define loss function (criterion) and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                         momentum=self.momentum, weight_decay=self.weight_decay)

        self._resume()
        self.train_loader, self.val_loader = self._data()

        pass

    def _create_model(self):
        # create model
        Tools.print("=> using model '{}'".format(self.arch))
        model = models.__dict__[self.arch](pretrained=self.has_pretrained)
        # DataParallel will divide and allocate batch_size to all available GPUs
        if self.arch.startswith('alexnet') or self.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
        return model.cuda()

    def _resume(self):
        if os.path.isfile(self.resume_filename):
            checkpoint = torch.load(self.resume_filename)
            self.start_epoch = checkpoint['epoch']
            self.best_acc1 = checkpoint['best_acc1'].cuda()  # best_acc1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            Tools.print("=> loaded checkpoint '{}' (epoch={}, acc={})".format(
                self.resume_filename, self.start_epoch, self.best_acc1))
        else:
            Tools.print("=> no checkpoint found at '{}'".format(self.resume_filename))

        cudnn.benchmark = True
        pass

    def _data(self):
        # Data loading code
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val_new')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=self.workers)

        test_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.workers)
        return train_loader, val_loader

    def _save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint_filename)
        if is_best:
            shutil.copyfile(self.checkpoint_filename, self.best_checkpoint_filename)
        pass

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 5))
        Tools.print("epoch={} lr={}".format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        pass

    @staticmethod
    def _accuracy(output, target, top_k=(1,)):
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

    def _train_one_epoch(self, epoch):
        batch_time, data_time = AverageMeter('Time', ':6.3f'), AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.train_loader),
                                 [batch_time, data_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch))
        self.model.train()

        def _each_iter(images, target):
            images = images.cuda()
            target = target.cuda()

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pass

        end = time.time()
        for i, (_images, _target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            end = time.time()

            _each_iter(_images, _target)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                progress.display(i)
                pass
            pass

        pass

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._adjust_learning_rate(epoch)

            self._train_one_epoch(epoch)

            acc1 = self.validate()
            self.best_acc1 = max(acc1, self.best_acc1)

            self._save_checkpoint({'epoch': epoch + 1, 'arch': self.arch,
                                   'state_dict': self.model.state_dict(), 'best_acc1': self.best_acc1,
                                   'optimizer': self.optimizer.state_dict()}, acc1 >= self.best_acc1)
            pass
        pass

    def validate(self):
        batch_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

        self.model.eval()
        with torch.no_grad():

            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.cuda()
                target = target.cuda()

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.display(i)
                    pass
                pass

            Tools.print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            pass

        return top1.avg

    pass


if __name__ == '__main__':
    RunnerSingle().train()
    pass
