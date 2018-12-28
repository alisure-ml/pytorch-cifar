from __future__ import print_function
import os
import torchvision
from models import *
import torch.nn as nn
import torch.optim as optim
from utils import print_info
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


class Runner(object):

    def __init__(self, root_path='/data/DATASET/cifar', model=VGG, batch_size=128, lr=0.1, name="vgg"):
        """
        # net = VGG()
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2()
        """
        self.root_path = root_path
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.checkpoint_path = "./checkpoint/{}".format(self.name)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = self._build(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_loader, self.test_loader = self._data()
        pass

    def _data(self):
        print_info('==> Preparing data..')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.CIFAR10(self.root_path, train=True, download=True, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=True, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        return _train_loader, _test_loader

    def _build(self, model):
        print_info('==> Building model..')
        net = model()

        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            pass
        return net

    def _change_lr(self, epoch):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < 100:
            __change_lr(self.lr)
        elif 100 <= epoch < 200:
            __change_lr(self.lr / 10)
        elif 200 <= epoch:
            __change_lr(self.lr / 100)

        pass

    def info(self):
        print_info("model={} batch size={} lr={} name={}".format(str(self.model), self.batch_size, self.lr, self.name))
        pass

    def resume(self, is_resume):
        if is_resume and os.path.isdir(self.checkpoint_path):
            print_info('==> Resuming from checkpoint..')
            checkpoint = torch.load('{}/ckpt.t7'.format(self.checkpoint_path))
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        pass

    def train(self, epoch, change_lr=False):
        print()
        print_info('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch)

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        print_info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss / len(self.train_loader), 100. * correct / total, correct, total))
        pass

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass
            pass

        print_info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (test_loss / len(self.test_loader), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print_info('Saving..')
            state = {'net': self.net.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            torch.save(state, '{}/ckpt.t7'.format(self.checkpoint_path))
            self.best_acc = acc
            pass
        print_info("best_acc={} acc={}".format(self.best_acc, acc))
        pass

    pass


if __name__ == '__main__':

    runner = Runner(model=MobileNetV2, batch_size=128, lr=0.01, name="MobileNetV2")
    runner.info()
    runner.resume(is_resume=True)

    for _epoch in range(runner.start_epoch, 300):
        runner.train(_epoch, change_lr=True)
        runner.test(_epoch)
        pass
    pass
