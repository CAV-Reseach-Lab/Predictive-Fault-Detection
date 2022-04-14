import cv2
from pytorch_wavelets import DWTForward
import numpy as np
import torchvision.transforms as transforms
import torch

path_dir = "D:/a2d2/camera_lidar-Gaimersheim\cam_front_center/"

inputs = []
for i in ["20180810150607_camera_frontcenter_000000083", "20180810150607_camera_frontcenter_000000084"]:
    dir = path_dir + i + ".png"
    img = cv2.cvtColor(cv2.imread(dir), cv2.COLOR_BGR2RGB)
    inputs.append(transforms.ToTensor()(img.copy()))
inputs = torch.stack(inputs, dim=0)
print("inputs:", np.shape(inputs))

tmp = inputs[0]
print("tmp:", np.shape(tmp))

dwt1 = DWTForward(J=1, wave='haar', mode='symmetric')

dwt1_1_l, dwt1_1_h = dwt1(inputs)
print("dwt1_1_l", np.shape(dwt1_1_l))
print("dwt1_1_h", np.shape(dwt1_1_h[0]))
print("dwt1_1_h[:,:,0]", np.shape(dwt1_1_h[0][:,:,0]))

#cv2.imshow(img)
#cv2.imshow(dwt1_1_l[0])
#cv2.imshow(dwt1_1_h[0])
#cv2.imshow(img)


