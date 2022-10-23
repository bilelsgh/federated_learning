# TensorFlow and tf.keras
from cgi import test
from gc import callbacks
from msilib.schema import File
import tensorflow as tf
from tensorflow.keras import models, layers

from matplotlib import pyplot as plt
import os
import numpy as np
from threading import Lock
from utils.format_dataset import create_datasets_grocery, create_datasets_mnist


class Model :

    def __init__(self,checkpoint_path,labels,data_path=None):
        self.model = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.labels = labels
        self.lock = Lock()

        self.create_model()

        self.load_data()


    def load_data(self):
        if self.data_path:
            # create_datasets_grocery(self.data_path,self.labels, "datasets/preprocessed_homemade_grocery_dataset") 
            create_datasets_mnist(self.data_path,self.labels) 

            self.train_images = np.load(f'{self.data_path}/data/train_images.npy')
            self.train_labels = np.load(f'{self.data_path}/data/train_labels.npy')
            self.test_images = np.load(f'{self.data_path}/data/test_images.npy')
            self.test_labels = np.load(f'{self.data_path}/data/test_labels.npy')
        else:
            exit("Please indicate a dataset path")

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # 32 filters
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(64, (3, 3), activation='relu')) # 64 filters
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(64, (3, 3), activation='relu')) # 64 filters

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

        # Need to compile the model before using it
        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        #self.save_weights()
        
    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5) # change 
        self.save_weights()


    def eval(self):        
        test_loss, test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print('\n# Eval :\n - Labels: {}\n - Accuracy: {}'.format(self.labels,test_acc)) 
        return test_acc

    def save_weights(self):
        for file in os.listdir(f"{self.checkpoint_path}/saved"):
            if file in ["cp.ckpt.data-00000-of-00001","cp.ckpt.index","checkpoint"] :
                os.remove(f"{self.checkpoint_path}/saved/{file}")

        with open(f"{self.checkpoint_path}/saved/labels.txt","w") as f:
            f.write("/".join( list( map( lambda x: str(x), self.labels )) ) ) 

        self.model.save_weights(f"{self.checkpoint_path}/saved/cp.ckpt")

        
    
    def average_model(self,models):
        print(f" (model) About to merge {len(models)}")
        new_weights = list()
        weights = [model_.get_weights() for model_ in models]

        for idx, weights_list_tuple in enumerate(zip(*weights)): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

        self.model.set_weights(new_weights)
        self.save_weights()

    
    def visualize_filters(self):
        """
        Visualize filters used in this model
        Source: https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
        """

        res = list()

        for layer in self.model.layers:
            if 'conv' in layer.name:
                weights, bias= layer.get_weights()
                res.append(weights)

                #normalize filter values between  0 and 1 for visualization
                f_min, f_max = weights.min(), weights.max()
                filters = (weights - f_min) / (f_max - f_min)  
                filter_cnt=1
                
                #plotting all the filters
                for i in range(filters.shape[3]):
                    #get the filters
                    filt=filters[:,:,:, i]
                    #plotting each of the channel, color image RGB channels
                    for j in range(filters.shape[0]):
                        ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.imshow(filt[:,:, j])
                        filter_cnt+=1
               # plt.show()

        return res
       
    # Setters
    def set_labels_range(self,labels):
        self.labels = labels
        self.load_data()

        return set(self.train_labels)

    def load_model(self):
        self.model.load_weights(f"{self.checkpoint_path}/received/cp.ckpt")

    def perf_load_model(self):
        self.model.load_weights(f"{self.checkpoint_path}/saved/cp.ckpt")

    # Getters
    def get_weights(self):
        return self.model.get_weights() # return filters AND biases
    
    def get_size_dataset(self):
        return len(self.train_images)

    def get_labels(self):
        return self.labels

    def get_save_path(self):
        return f"{self.checkpoint_path}/saved"
    
    # Lock
    def acLock(self):
        self.lock.acquire()
    
    def reLock(self):
        self.lock.release()

    def predict(self,img):
        probability_model = tf.keras.Sequential([self.model, 
                                         tf.keras.layers.Softmax()])
        prediction = probability_model.predict(img)

        return prediction

