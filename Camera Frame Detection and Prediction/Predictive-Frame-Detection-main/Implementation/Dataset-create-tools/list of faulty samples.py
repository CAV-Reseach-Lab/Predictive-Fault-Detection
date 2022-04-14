import pickle

city = "Inglostadt"
parent_path = '/home/server00/Sampo/Datasets/A2D2/camera_lidar-' + city

fault_samples = []
for i in ["/rainy_samples_list_", "/snowy_samples_list_", "/foggy_samples_list_"]:
    pickle_file = parent_path + i + city
    with open(pickle_file, 'rb') as handle:
        a = pickle.load(handle)
    #fault_samples.append(a[::])
    fault_samples = fault_samples + a

nums = []
for str in fault_samples:
    tmp = str[len(parent_path + "/front_center_camera_resample/"):]
    nums.append(int(tmp))

fault_samples = sorted(fault_samples)
nums = sorted(nums)
print("size of list:", len(fault_samples))
print(fault_samples)
print(nums)
