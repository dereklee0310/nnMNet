import json
import random
from pathlib import Path

import natsort as ns

random.seed(42069)

# Split is available and in nature sorted order, so we only need to split it
files = ns.natsorted(x.stem for x in Path("SynMars-TW_V2/imagesTr").glob("*"))
# train: 17000 / val: 2000 / train + val: 19000
train, val = files[:17000], files[17000:]
print("train len: ", len(train), "val len: ", len(val))
with open("./nnUNet_preprocessed/Dataset777_SynMars-TW/splits_final.json", "w+") as f:
    data = [{"train": train, "val": val}]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

# Split is not available, shuffle it
files = ns.natsorted(x.stem for x in Path("MarsScapes_V2/imagesTr").glob("*"))
random.shuffle(files)
train, val = files[:13002], files[13002:]
print("train len: ", len(train), "val len: ", len(val))
with open("./nnUNet_preprocessed/Dataset778_MarsScapes/splits_final.json", "w+") as f:
    data = [{"train": train, "val": val}]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()


files = ns.natsorted(x.stem for x in Path("SynMars-Air_V2/imagesTr").glob("*"))
# 9400, 1150, 1150
train, val = [], []
for file in files:
    if file.endswith("train"):
        train.append(file)
    else:
        val.append(file)

print("train len: ", len(train), "val len: ", len(val))
with open("./nnUNet_preprocessed/Dataset779_SynMars-Air/splits_final.json", "w+") as f:
    data = [{"train": train, "val": val}]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()


# Split is available but not reliable, shuffle it
files = ns.natsorted(x.stem for x in Path("S5Mars_V2/imagesTr").glob("*"))
random.shuffle(files)
train, val = files[:5000], files[5000:]
print("train len: ", len(train), "val len: ", len(val))
with open("./nnUNet_preprocessed/Dataset781_S5Mars/splits_final.json", "w+") as f:
    data = [{"train": train, "val": val}]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()


# MSL-Seg, deprecated
files = ns.natsorted(x.stem for x in Path("MSL-Seg_V2/imagesTr").glob("*"))
random.shuffle(files)
train, val = files[:2480], files[2480:]
print("train len: ", len(train), "val len: ", len(val))
with open("./nnUNet_preprocessed/Dataset780_MSL-Seg/splits_final.json", "w+") as f:
    data = [{"train": train, "val": val}]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()


# ---------------------------------------------------------------------------- #
#                                  Old Splits                                  #
# ---------------------------------------------------------------------------- #
# import json
# from batchgenerators.utilities.file_and_folder_operations import load_json

# SPLIT_FILE = "./nnUNet_preprocessed/Dataset777_SynMars-TW/splits_final.json"
# NUM_TRAIN = 17000
# # train(15200)/val(3800) x 5 -> train(17000)/val(2000)
# splits = load_json(SPLIT_FILE)
# # print(len(splits))
# full_dataset = splits[0]["train"] + splits[0]["val"]
# train = full_dataset[:NUM_TRAIN]
# val = full_dataset[NUM_TRAIN:]
# print(len(train), len(val))
# with open(SPLIT_FILE, "w") as f:
#     json.dump([{"train": train, "val": val}], f, indent=4)

# SPLIT_FILE = "./nnUNet_preprocessed/Dataset778_MarsScapes/splits_final.json"
# NUM_TRAIN = 13002 # 17194 - 4192
# splits = load_json(SPLIT_FILE)
# # print(len(splits))
# full_dataset = splits[0]["train"] + splits[0]["val"]
# train = full_dataset[:NUM_TRAIN]
# val = full_dataset[NUM_TRAIN:]
# print(len(train), len(val))
# with open(SPLIT_FILE, "w") as f:
#     json.dump([{"train": train, "val": val}], f, indent=4)
