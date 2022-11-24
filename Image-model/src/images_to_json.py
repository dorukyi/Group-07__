# import required module
import os
import json

# assign directory
directory = 'Datasets/BIRDS450/images'
image_dict = dict()
data_types = []
index_dict = dict()
# iterate over files in
# that directory
for data_type in os.listdir(directory):
    data_types.append(data_type)
    
for indx, bird_name in enumerate(os.listdir(os.path.join(directory, data_types[0]))):
    index_dict[indx] = bird_name


for type in data_types:
    f = os.path.join(directory, type)
    mid_dict = dict()
    for bird_name in os.listdir(f):
        images = os.path.join(f, bird_name)
        lst = []
        for number in os.listdir(images):
            lst.append(number)
        mid_dict[bird_name] = lst
    image_dict[type] = mid_dict

with open("src/all_image_files.json", "w") as outfile:
    json.dump(image_dict, outfile)

with open("src/index_to_class_label.json", "w") as outfile:
    json.dump(index_dict, outfile)
