import numpy as np
import sys
import shutil
import os
from dotenv import load_dotenv
from PIL import Image 
import cv2
import tensorflow as tf

load_dotenv('.env')
DATASET_PATH = os.getenv('DATASET_PATH')
SIZE = int(os.getenv('SIZE'))


def display_datasets():
    files = ["train_labels","train_images","test_labels","test_images"]

    for i in range(1,4):
        print(f"\n// DEVICE #{i}")
        for file in files:
            try:
                tmp = np.load(f'device/device{i}/data/{file}.npy')
                print(f"* {file}\n",tmp.shape)
                if 'labels' in file:
                    print(np.unique(tmp))

            except:
                print(f"{file} doesn't exist.")
        print("-----------")



def create_datasets_grocery(device, labels, data_path):
    """
    Create a dataset containing only some labels from grocery dataset
    :param device: Device folder - str
    :param labels: Label(s) of the class(es) that should be present in the dataset - <str>[]
    """


    print("(format_dataset.py) About to generate the datasets")
    train_images, train_labels = get_images_labels(f"{data_path}/train")
    test_images, test_labels = get_images_labels(f"{data_path}/test")
    data = [ (train_images, train_labels), (test_images, test_labels) ]

    for idx,images_labels in enumerate(data):
        dataset_images = []
        dataset_labels = []
        for img,label in zip(images_labels[0],images_labels[1]):
            if label in labels:
                dataset_images.append(img)
                dataset_labels.append(label)
        
        step = "train" if idx == 0 else "test"
        np.save(f'{device}/data/{step}_images', dataset_images)
        np.save(f'{device}/data/{step}_labels', dataset_labels)



def create_datasets_mnist(device, labels):
    """
    Create a dataset containing only some labels from mnist dataset
    :param device: Device folder - str
    :param labels: Label(s) of the class(es) that should be present in the dataset - <str>[]
    """

    print("(format_dataset.py) About to generate the datasets")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    data = [ (train_images, train_labels), (test_images, test_labels) ]

    for idx,images_labels in enumerate(data):
        dataset_images = []
        dataset_labels = []
        for img,label in zip(images_labels[0],images_labels[1]):
            if label in labels:
                dataset_images.append(img)
                dataset_labels.append(label)
        
        step = "train" if idx == 0 else "test"
        np.save(f'{device}/data/{step}_images', dataset_images)
        np.save(f'{device}/data/{step}_labels', dataset_labels)

        

def preprocess_images(current_dataset_path, new_datatset_path):
    """
    Create a new dataset where every image has the same dimension.
    """

    # Explore the current dataset
    for class_ in os.listdir(current_dataset_path):
        for img_ in os.listdir(current_dataset_path + class_):

            image =  Image.open(f"{current_dataset_path}{class_}/{img_}")
            size = image.size[0] if image.size[0] > image.size[1] else image.size[1] 

            # We paste the current image on a square white image with side = current image's greater dimension 
            blank = Image.new('RGBA', (size,size), (255, 255, 255) )
            coord = (0, ( size - min(image.size) )//2 ) if image.size[0] > image.size[1] else ( ( size - min(image.size) )//2, 0 )
            blank.paste(image, coord)
            to_save = blank.resize( (SIZE,SIZE) )

            # We save the new square image
            try:
                to_save.save(f"{new_datatset_path}/raw/{class_}/{img_.split('.')[0]}.png")
            except:
                os.mkdir(f"{new_datatset_path}/raw/{class_}")
                to_save.save(f"{new_datatset_path}/raw/{class_}/{img_.split('.')[0]}.png")



def split_dataset(dataset_path, prop_train ):
    """
    Split a raw dataset into two datasets : one for train and one for test
    """

    # Explore the whole dataset 
    for class_ in os.listdir(f"{dataset_path}/raw"):
        if class_ not in ["train", "test"]:
            size = len(os.listdir( f"{dataset_path}/raw/{class_}" ))
            for idx, img_ in enumerate( os.listdir(f"{dataset_path}/raw/{class_}") ): 

                # Define if an image belong to the train dataset or the test datatset according to the proportion asked
                dest_path = f"{dataset_path}/train/{class_}" if idx < round(size*prop_train) else f"{dataset_path}/test/{class_}"
                
                # Copy
                try:
                    shutil.copyfile( f"{dataset_path}/raw/{class_}/{img_}", f"{dest_path}/{img_}" )
                except:
                    os.mkdir(dest_path)
                    shutil.copyfile( f"{dataset_path}/raw/{class_}/{img_}", f"{dest_path}/{img_}" )


def get_images_labels(dataset_path):
    """
    From a dataset get an array containing images and another containing labels
    :param dataset_path: Path pointing to the dataset containing every class
    :return: Two np arrays images and labels
    """

    images = []
    labels = []
    label_to_int = {"Apple": 0, "Banana": 1, "Chips": 2, "Candy": 3, "Car": 4, "Pasta": 5, "Yogurt": 6}

    for class_ in os.listdir(dataset_path):
        for img in os.listdir(f"{dataset_path}/{class_}"):
            images.append( cv2.imread(f"{dataset_path}/{class_}/{img}") )
            labels.append(label_to_int[class_])

    return np.array( images,  dtype="float"), np.array( labels,  dtype="float")


def test():
    exit("nothing there so far")




if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("Please call a function among these ones : display_datasets, test, preprocess")

    userArg = sys.argv[1]

    if userArg == "display_datasets":
        display_datasets()

    elif userArg == "test":
        test()

    elif userArg == "preprocess":
        preprocess_images(DATASET_PATH, "datasets/preprocessed_homemade_grocery_dataset")

    elif userArg == "split_dataset":
        split_dataset("datasets/preprocessed_homemade_grocery_dataset", 0.8)
    
    else:
        exit("This function doesn't exist. Please chose one of these functions :\n - display_datasets\n - test\n - preprocess")

