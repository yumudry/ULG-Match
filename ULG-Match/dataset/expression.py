import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from .dataset import AgriculturalDisease

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
AgriculturalDisease_mean=(0.4608,0.4858,0.3959)
AgriculturalDisease_std=(0.1948,0.1766,0.2154)
PlantVillage_mean=(0.4598,0.5710,0.4269)
PlantVillage_std=(0.2148,0.2222,0.2208)
apple_mean=(0.5663,0.5965,0.5217)
apple_std=(0.2517,0.2400,0.2881)
raf_mean=(0.485, 0.456, 0.406)
raf_std=(0.229, 0.224, 0.225) 


def get_raf(args, root):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,
                              padding=int(224*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=raf_mean, std=raf_std)#修改归一化操作
    ])
    transform_val = transforms.Compose([
        # transforms.RandomCrop(size=32,
        #             padding=int(32*0.125),
        #             padding_mode='reflect'),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,
                              padding=int(224*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=raf_mean, std=raf_std)
    ])
    base_dataset = AgriculturalDisease(root)
    print(base_dataset.class_to_index)
    train_labeled_idxs, train_unlabeled_idxs = split_data(
         args, base_dataset.labels)
    # print(train_labeled_idxs.shape)
    train_labeled_dataset = ARGSSL(
        root, train_labeled_idxs,
        transform=transform_labeled)
    train_unlabeled_dataset = ARGSSL(
        root, train_unlabeled_idxs,
        transform=TransformFixMatch(mean=raf_mean, std=raf_std))#修改归一化操作
    test_dataset = AgriculturalDisease(data_folder='/home/user-lbrhk/dataset/RAF/valid/',
                                                transform=transform_val)
 
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def split_data(args,labels):
    labels = np.array(labels)
    print(labels.shape)
    labeled_idx = []
    ratio = 0.01
    print(ratio)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        # 计算 10% 的数量
        num_labeled_per_class = int(ratio * len(idx))
        # 从每个类别的数据中随机选择 10% 作为有标签数据
        labeled_idx.extend(np.random.choice(idx, num_labeled_per_class, replace=False))
    labeled_idx = np.array(labeled_idx)

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        # print(len(idx))
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    print(len(labeled_idx))
    # assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,#原本为32
                              padding=int(224*0.125),
                              padding_mode='reflect'),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,#原本为32
                              padding=int(224*0.125),
                              padding_mode='reflect'),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])#有修改
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
def default_loader(path):
    return Image.open(path).convert('RGB')

class ARGSSL(AgriculturalDisease):
    def __init__(self, data_folder, indexs,transform=None, target_transform=None,loader=default_loader):
        super().__init__(data_folder,transform=transform,
                         target_transform=target_transform,loader=default_loader)
        if indexs is not None:
            self.image_paths = np.array(self.image_paths)[indexs]
            self.labels = np.array(self.labels)[indexs]

        #  # 打印类别名和标签的对应关系
        # print("Class to Index Mapping:")
        # for class_name, label in self.class_to_index.items():
        #     print(f"Class: {class_name}  |  Label: {label}")



    def __getitem__(self, index):
        image_path, label = self.image_paths[index], self.labels[index]
        # image = Image.open(image_path).convert('RGB')
        # image = Image.fromarray(image)
        image=self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'RAF-DB':get_raf}
