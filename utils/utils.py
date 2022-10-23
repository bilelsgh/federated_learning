import os
import numpy as np


def get_pointeur(device_nb):
    pointeur = -1
    with open("pointeurs.txt","r") as f:
        line = f.readline().split("_")
        if line == "":
            return -1 

        while line[0] != f"DEVICE{device_nb}" :
            line = f.readline().split("_")
        pointeur = line[1]

    return int(pointeur)



def check_dir(device):
    dir = os.listdir(f"server_training/{device}/received")
    file_list = ["cp.ckpt.data-00000-of-00001","cp.ckpt.index"] 

    # We don't have the 2 files describing the weights yet
    print(f"\DIR {device}\\ {dir} \n\n")
    if len(dir) != 3 :
        return False

    # Check if the two files are the good ones
    for file in dir:
        if file != ".gitkeep" and file not in file_list:
            return False
    return True

def format_dataset(train_images, train_labels, labels):
    imgs = []
    lbl = []

    for img,label in zip(train_images,train_labels):
        if str(label) in labels:
            imgs.append(img)
            lbl.append(label)
        
    return np.array(imgs), np.array(lbl)


def get_labels(device):
    with open(f"server_training/{device}/labels.txt","r") as f:
        res = f.read().split("/")

    return list( map( lambda x: int(x), res) )