import os
import time
import torch
import random
import shutil
import cProfile
import torch.optim
from glob import glob
import multiprocessing
import torch.utils.data
import torch.nn.parallel
from functools import wraps
import torch.utils.data.distributed
from alisuretool.Tools import Tools
import torch.utils.data.distributed
import torchvision.models as models
from line_profiler import LineProfiler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
https://www.cnblogs.com/king-lps/p/10936374.html
https://blog.csdn.net/winycg/article/details/92443146
"""


class InputTrain(object):

    def __init__(self, data_root, batch_size, num_workers, queue_max_size):
        self.data_root = data_root
        self.batch_size = batch_size
        self.queue_max_size = queue_max_size
        self.queue = multiprocessing.Queue(maxsize=self.queue_max_size)

        self.train_dataset = self.data()

        self.num_workers = num_workers
        self.data_len = len(self.train_dataset.imgs)
        self.num_per_worker = self.data_len // self.num_workers

        self.split_data = []
        self.process = []
        pass

    def data(self):
        # train_dir = os.path.join(self.data_root, 'train')
        train_dir = os.path.join(self.data_root, 'val_new')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
        return train_dataset

    def init(self):
        img_index = list(range(self.data_len))
        random.shuffle(img_index)
        self.split_data = [img_index[i * self.num_per_worker:
                                     (i + 1) * self.num_per_worker] for i in range(self.num_workers)]
        pass

    def start_queue(self):
        self.process = [multiprocessing.Process(
            target=self._queue_put, args=(self.queue, self.split_data[i])) for i in range(self.num_workers)]
        [t.start() for t in self.process]
        pass

    def __getitem__(self, item):
        images, labels = self.queue.get()
        return images, labels

    def _queue_put(self, queue, index_list):
        batch_num = len(index_list) // self.batch_size
        for i in range(batch_num):
            images, labels = self._get_batch_data(index_list[i * self.batch_size: (i + 1) * self.batch_size])
            queue.put([images, labels])
            pass
        pass

    def _get_batch_data(self, indexes):
        assert len(indexes) == self.batch_size
        images, labels = [], []
        for i in indexes:
            sample, target = self.train_dataset.__getitem__(i)
            images.append(sample)
            labels.append(target)
            pass
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

    pass


class Runner(object):

    def __init__(self, train_loader):
        self.model = torch.nn.DataParallel(models.__dict__["resnet18"](pretrained=False)).cuda()

        self.train_loader = train_loader
        pass

    def train(self):
        self.model.train()

        time_sum = 0
        end = time.time()
        self.train_loader.init()
        self.train_loader.start_queue()
        for i, (_images, _target) in enumerate(self.train_loader):
            use_time = time.time() - end
            time_sum += use_time
            end = time.time()
            Tools.print("i={} use={:4f} avg={:4f} sum={:4f}".format(i, use_time, time_sum / (i + 1), time_sum))
            pass
        pass

    @staticmethod
    def demo():
        input_train = InputTrain(data_root="/home/z840/data/DATASET/ILSVRC2015/Data/CLS-LOC",
                                 batch_size=256, num_workers=30, queue_max_size=100)
        Runner(train_loader=input_train).train()
        pass

    pass


def copy_data():
    replace1 = "/data/DATASET/ILSVRC2015/Data/CLS-LOC"
    replace2 = "/media/z840/ALISURE/data/DATASET/ILSVRC2015/Data/CLS-LOC"

    image_file_list = glob("{}/train/*/*.JPEG".format(replace1))

    for index, image_file in enumerate(image_file_list):
        if index % 10000 == 0:
            print("{} {}".format(index, len(image_file_list)))
        image_file2 = image_file.replace(replace1, replace2)
        if not os.path.exists(image_file2):
            print(image_file2)
            shutil.copy(image_file, image_file2)
            pass
        pass
    pass


class TimeStat(object):

    @staticmethod
    def func_time(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print(f.__name__, 'took', end - start, 'seconds')
            return result

        return wrapper

    @staticmethod
    def func_profile(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = f(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats(sort='time')

        return wrapper

    @staticmethod
    def func_line_time(follow=[]):
        def decorate(func):
            @wraps(func)
            def profiled_func(*args, **kwargs):
                profiler = LineProfiler()
                try:
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
                pass
            return profiled_func
        return decorate

    pass


@TimeStat.func_line_time()
def first():
    for x in range(1000000):
        if x % 100000 == 0:
            print(x)
    pass


if __name__ == '__main__':
    first()
    pass
