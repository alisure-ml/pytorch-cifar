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


# 转换数据：val -> val_new
class TranTiny(object):

    def __init__(self, val_root="/home/z840/ALISURE/Data/tiny-imagenet-200/val",
                 tiny_val_txt="val_annotations.txt",
                 val_result_root="/home/z840/ALISURE/Data/tiny-imagenet-200/val_new"):
        self.val_root = val_root
        self.val_result_root = val_result_root
        self.tiny_val_txt = os.path.join(self.val_root, tiny_val_txt)
        self.tiny_val_image_path = os.path.join(self.val_root, "images")

        self.val_data = self.read_txt()
        pass

    def read_txt(self):
        with open(self.tiny_val_txt) as f:
            tine_val = f.readlines()
            return [i.strip().split("\t")[0:2] for i in tine_val]
            pass
        pass

    def new_val(self):
        for index, (image_name, image_class) in enumerate(self.val_data):
            if index % 100 == 0:
                Tools.print("{} {}".format(index, len(self.val_data)))
            src = os.path.join(self.tiny_val_image_path, image_name)
            dst = Tools.new_dir(os.path.join(self.val_result_root, image_class, image_name))
            shutil.copy(src, dst)
            pass
        pass

    @staticmethod
    def main():
        TranTiny().new_val()
        pass

    pass


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


class RunnerSingle(object):

    def __init__(self, lr=0.1, print_freq=10, start_epoch=0, epochs=90, output_size=64, num_classes=200,
                 batch_size=256, workers=30, momentum=0.9, weight_decay=1e-4, arch="resnet18",
                 data_root="/home/z840/ALISURE/Data/tiny-imagenet-200",
                 resume_filename="./checkpoint_tiny_imagenet/ResNet18/checkpoint.pth.tar",
                 checkpoint_filename="./checkpoint_tiny_imagenet/ResNet18/checkpoint.pth.tar",
                 best_checkpoint_filename="./checkpoint_tiny_imagenet/ResNet18/checkpoint_best.pth.tar"):
        self.resume_filename = resume_filename
        self.checkpoint_filename = Tools.new_dir(checkpoint_filename)
        self.best_checkpoint_filename = best_checkpoint_filename

        self.lr = lr
        self.output_size = output_size
        self.num_classes = num_classes
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

        self.train_loader, self.val_loader = self._data()

        self.model = self._create_model(self.num_classes)
        self.criterion = nn.CrossEntropyLoss().cuda()  # define loss function (criterion) and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                         momentum=self.momentum, weight_decay=self.weight_decay)

        self._resume()
        pass

    def _data(self):
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val_new')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([transforms.RandomResizedCrop(self.output_size),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.Resize(self.output_size),
                                             transforms.CenterCrop(self.output_size), transforms.ToTensor(), normalize])

        train_dataset = datasets.ImageFolder(train_dir, transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=self.workers)

        test_dataset = datasets.ImageFolder(val_dir, transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.workers)
        return train_loader, val_loader

    def _create_model(self, num_classes):
        # create model
        Tools.print("=> using model '{}'".format(self.arch))
        model = models.__dict__[self.arch](pretrained=self.has_pretrained, num_classes=num_classes)

        if "resnet" in self.arch:
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            pass

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

    def _save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint_filename)
        if is_best:
            shutil.copyfile(self.checkpoint_filename, self.best_checkpoint_filename)
        pass

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
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


def main():
    _data_root = "/home/z840/ALISURE/Data/tiny-imagenet-200"
    _arg = ["resnet18", 0.1, 256, 28, "resnet18", 64]
    # _arg = ["resnet18", 0.1, 256, 28, "resnet18_3", 224]
    RunnerSingle(lr=_arg[1], print_freq=50, start_epoch=0, epochs=90,
                 batch_size=_arg[2], workers=_arg[3], momentum=0.9, output_size=_arg[5], num_classes=200,
                 weight_decay=1e-4, arch=_arg[0], data_root=_data_root,
                 resume_filename="./checkpoint_tiny_imagenet/{}/checkpoint.pth.tar".format(_arg[4]),
                 checkpoint_filename="./checkpoint_tiny_imagenet/{}/checkpoint.pth.tar".format(_arg[4]),
                 best_checkpoint_filename="./checkpoint_tiny_imagenet/{}/checkpoint_best.pth.tar".format(_arg[4])
                 ).train()
    pass


if __name__ == '__main__':
    main()
    pass
