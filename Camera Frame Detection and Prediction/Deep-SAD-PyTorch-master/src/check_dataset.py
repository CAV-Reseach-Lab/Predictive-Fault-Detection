import pickle
import numpy as np

class dataset:
    def __init__(self, data):
        self.targets = data[1]
        self.feature = data[0]
        self.semi_targets = np.zeros_like(self.targets)

    def __getitem__(self, item):
        img, target, semi_target = self.feature[item], self.targets[item], self.semi_targets[item]
        return img, target, semi_target, item
    def __len__(self):
        return len(self.feature)

with open('train.pt', 'rb') as file:
    #data = dataset
    data = pickle.load(file)
    print(len(data))

