
import os
import json
import pandas as pd

index_dict = {}

df = pd.read_csv("dataset.csv")
labels = df.iloc[:,-1].unique()
labels.sort()

for indx, bird_name in enumerate(labels):
    index_dict[indx] = bird_name

with open("src/index_to_class_label.json", "w") as outfile:
    json.dump(index_dict, outfile)
