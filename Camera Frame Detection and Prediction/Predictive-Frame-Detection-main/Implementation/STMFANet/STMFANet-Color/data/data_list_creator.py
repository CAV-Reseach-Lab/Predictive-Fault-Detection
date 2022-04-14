currentline = []
with open("./augmented_samples.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(", ")
print(currentline)


samples_index = []
for num in currentline:
    samples_index.append(int(num))
print(samples_index)


samples_address = []
path = "/home/server00/Sampo/Datasets/A2D2/camera_lidar-Inglostadt/front_center_camera_resample/"
for i in range(1, 228 + 1):
    if i not in samples_index:
        samples_address.append(path + str(i) + " 0 119")
    else:
        samples_address.append(path + str(i) + "/anomaly" + " 0 119")

with open("validation_data_list.txt", "w") as filestream_t:
    for item in samples_address:
        filestream_t.write( str(item + "\n"))
