#the detector net object to detect faults
import torch
import time
import numpy as np
from scipy.signal import savgol_filter
import libs.ince


class detectorNet():
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('./libs/detectorV3.pt', self.device)

    def HIextract(self, prediction):
        window = 1
        poly = 0
        healthy = savgol_filter(prediction[0:, 0], window, poly) + 22.5
        faulty = savgol_filter(prediction[0:, 1], window, poly) - 22.5
        diff = healthy - faulty
        #hi = savgol_filter(diff, window, poly)
        hi = diff
        #print(hi)

        return hi[0]

    def predict(self, dataFrame):
        with torch.no_grad():
            dataFrame = dataFrame.to(device=self.device, dtype=torch.float)
            prediction = self.model(dataFrame)
            if torch.cuda.is_available():
                predCPU = prediction.cpu()
                result = np.argmax(predCPU)
                hi = self.HIextract(predCPU)
            else:
                result = np.argmax(prediction)
                hi = self.HIextract(prediction)
            result = result.numpy()
        return result, hi
