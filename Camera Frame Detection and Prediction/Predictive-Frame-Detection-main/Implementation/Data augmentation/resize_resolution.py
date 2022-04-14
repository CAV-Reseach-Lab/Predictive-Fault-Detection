import cv2
import glob
from tqdm import tqdm

#rescales the fully augmented images down for the prediction algorithm
#NOTE: MAKE SURE THE NEW RESOLUTION IS DIVISIBLE BY 8 OR ELSE THE PREDICTION ALGORITHM WONT WORK.

datapath = '/media/server00/sda/Sampo/Datasets/A2D2/camera_lidar-Munich/front_center_camera_resample'
imageFolders = 229
scalePercent = 25

imageList = []
anomalyList = []


def rescale(imageList):
    for folder in tqdm(imageList):
        for image in folder:
            src = cv2.imread(image, cv2.IMREAD_UNCHANGED)

            width = int(src.shape[1] * scalePercent / 100)
            height = int(src.shape[0] * scalePercent / 100)

            output = cv2.resize(src, (width, height))

            cv2.imwrite(image, output)


for i in range(imageFolders):
    imageList.append(glob.glob(datapath + '/' + str(i) + '/*.png'))

for i in range(imageFolders):
    anomalyList.append(glob.glob(datapath + '/' + str(i) + '/anomaly/*.png'))

for folder in anomalyList:
    if len(folder) == 0:
        anomalyList.remove(folder)

rescale(imageList)
rescale(anomalyList)


