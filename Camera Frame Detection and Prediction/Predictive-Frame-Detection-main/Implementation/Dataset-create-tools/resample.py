import os
import glob
import math
import shutil

parentPath = "/home/server00/Sampo/Datasets/A2D2/camera_lidar-"
sample_size = 100
#cityCode = {"Munich": "20190401_121727", "Gaimersheim": "20180810_150607", "Ingolstadt": "20190401_145936"}

def create_dest_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("folder created:", path)
        return
    print("There is a folder named:", path)
    return


def load_images(data_path):
    image_list = glob.glob(data_path + '/*.png')
    json_list = glob.glob(data_path + '/*.json')
    print(len(image_list), "frames loaded.", len(json_list), "json files loaded.")
    return image_list, json_list


def calc_number_of_samples(image_list):
    folder_numbers = math.floor(len(image_list) / sample_size)
    return int(folder_numbers)


def main():
    for city in ["Ingolstadt", "Gaimersheim"]:
        print("\n====================================== City:", city)
        dataset_path = parentPath + city
        create_dest_folder(dataset_path + "/front_center_camera_resample")

        data_path = dataset_path + "/cam_front_center"
        image_list, json_list = load_images(data_path)
        number_of_samples = calc_number_of_samples(image_list)

        for i in range(number_of_samples):
            destination = dataset_path + "/front_center_camera" + "/" + str(i+1)
            create_dest_folder(destination)
            for frame in range(i*sample_size, (i+1)*sample_size):
                shutil.copy2(image_list[frame], destination)
                shutil.copy2(json_list[frame], destination)
    print("All frames moved successfully.")

main()
