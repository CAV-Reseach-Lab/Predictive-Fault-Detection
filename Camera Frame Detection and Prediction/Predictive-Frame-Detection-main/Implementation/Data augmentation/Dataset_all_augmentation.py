import imgaug.augmenters as iaa
import math
import glob
import pickle
import os.path
import random
import cv2
import Automold as am

anomaly_percent_of_dataset = 100
city = "Inglostadt"
parent_path = '../Sampo/Datasets/A2D2/camera_lidar-' + city


def read_fault_samples_list(name):
    if os.path.isfile(name):
        with open(name, 'rb') as f:
            list_ = pickle.load(f)
        return list_
    return []


def select_candidates(anomaly_type, rainy_list, snowy_list, foggy_list):
    reserved_folders = rainy_list + snowy_list + foggy_list
    folders = glob.glob(parent_path + '/front_center_camera/*')
    rand_n = math.floor(len(folders) * anomaly_percent_of_dataset / 100)
    candidates_list = [x for x in folders if x not in (reserved_folders)]
    print("Candidates list:", candidates_list)
    if rand_n > len(candidates_list):
        raise Exception('You desired percentage for making anomaly samples is greater than the number of remaining normal samples in the dataset!')
    else:
        randomly_selected_folders = random.sample(candidates_list, rand_n)

    if anomaly_type == "rainy":
        with open(parent_path + "/rainy_samples_list_" + city, 'wb') as f:
            pickle.dump(rainy_list + randomly_selected_folders, f)
    elif anomaly_type == "snowy":
        with open(parent_path + "/snowy_samples_list_" + city, 'wb') as f:
            pickle.dump(snowy_list + randomly_selected_folders, f)
    elif anomaly_type == "foggy":
        with open(parent_path + "/foggy_samples_list_" + city, 'wb') as f:
            pickle.dump(foggy_list + randomly_selected_folders, f)
    else:
        raise Exception('The anomaly_type is not selected properly')
    return randomly_selected_folders


def make_anomaly_dir(save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print("created folder: ", save_path)
    else:
        print(save_path, "already exists!!")
    return


def load_images_from_folder(path):
    image_list =[]
    images = glob.glob(path)
    print('length of read images:', len(images))
    for index in range(len(images)):
        img = cv2.imread(images[index])
        image_list.append(img)
    return image_list


def add_rain(image_list):
    aug_images = []
    iter = range(1,len(image_list))
    print("Rainy Augmented image: ", end='', flush=True)
    for i in iter:
        rain = iaa.RainLayer(density=0.020 + 0.80/len(iter)*i, density_uniformity=0.5, drop_size=0.75 + 0.15/len(iter)*i, drop_size_uniformity=0.8,
                             angle=(12, 15), speed=0.2, blur_sigma_fraction=0.8,
                             seed=None, name=None, random_state="deprecated", deterministic="deprecated")

        aug_images = aug_images + rain(images=[image_list[i]])
        print(",", i, end='', flush=True)
    print()
    aug_images = [image_list[0]] + aug_images  # add the first image
    return aug_images


def add_snow(image_list):
    aug_images = []
    iter = range(1,len(image_list))
    print("Snowy Augmented image: ", end='', flush=True)
    for i in iter:
        snow = iaa.SnowflakesLayer(density=0.001 + 0.65/len(iter)*i, density_uniformity=0.7, flake_size=0.6 + 0.1/len(iter)*i, flake_size_uniformity=0.7,
                                   angle=(10,12), speed=0.01, blur_sigma_fraction=0,
                                   seed=None, name=None, random_state="deprecated", deterministic="deprecated")
        aug_images = aug_images + snow(images=[image_list[i]])
        print(",", i, end='', flush=True)
    print()
    aug_images = [image_list[0]] + aug_images  # add the first image
    return aug_images


def add_fog(image_list):
    aug_images = []
    aug_images.append([image_list[0]])
    iter = range(1, len(image_list))
    print("Foggy Augmented image: ", end='', flush=True)
    for i in iter:
        aug_images.append(am.add_fog([image_list[i]], fog_coeff=1/len(image_list)*i))
        print(",", i, end='', flush=True)
    print()
    return aug_images


def write_faulty_images(anomaly_type, save_path, images):
    print("Written images: ", end='', flush=True)
    for i in range(len(images)):
        if anomaly_type == "foggy":
            cv2.imwrite(save_path + '/' + str(i) + '.png', images[i][0])
        else:
            cv2.imwrite(save_path + '/' + str(i) + '.png', images[i])
        print(",", i, end='', flush=True)
    print()
    return


def main():

    #for anomaly_type in ["rainy", "snowy", "foggy"]:
    for anomaly_type in ["foggy"]:
        print("Anoamly Type: ", anomaly_type)
        foggy_list = read_fault_samples_list(parent_path + "/foggy_samples_list_" + city)
        snowy_list = read_fault_samples_list(parent_path + "/snowy_samples_list_" + city)
        rainy_list = read_fault_samples_list(parent_path + "/rainy_samples_list_" + city)
        randomly_selected_folders = select_candidates(anomaly_type, rainy_list, snowy_list, foggy_list)
        print("selected folders to make anomaly:", randomly_selected_folders)
        print("--------------------------------")

        for new_sample_folder in randomly_selected_folders:
            print(new_sample_folder)
            save_path = new_sample_folder + '/anomaly'
            make_anomaly_dir(save_path)
            images = load_images_from_folder(new_sample_folder + '/*.png')

            if anomaly_type == "rainy":
                aug_images = add_rain(images)
            elif anomaly_type == "snowy":
                aug_images = add_snow(images)
            elif anomaly_type == "foggy":
                aug_images = add_fog(images)
            else:
                raise Exception('The anomaly_type is not selected properly')
            write_faulty_images(anomaly_type, save_path, aug_images)
            print("\n")

        print("\n=========================================================")

main()
