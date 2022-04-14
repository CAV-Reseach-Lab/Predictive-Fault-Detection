from torch.utils.data import Subset
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import numpy as np
import pickle
import random

class a2d2Dataset(TorchvisionDataset):
    def __init__(self, data, normal_class, known_outlier_class, n_known_outlier_classes, ratio_known_normal, ratio_known_outlier
                 , ratio_pollution):
        super().__init__(data)

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)
        #self.test_set = dataset

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        #probably no preprocessing required

        with open(data + '/train.pt', 'rb') as file:
            #self.train_set = dataset([0, 0])
            self.train_set = pickle.load(file)

        #train_set = DataLoader(train_dataset, 8, True)

        idx, _, semi_targets = create_semisupervised_setting(self.train_set.targets, self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        self.train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        self.train_set = Subset(self.train_set, idx)

        with open(data + '/test.pt', 'rb') as file:
            self.test_set = pickle.load(file)

        #self.test_set = DataLoader(test_dataset, 8, False)

