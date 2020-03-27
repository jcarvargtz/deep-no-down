# :)
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, videos_foler_path, meta,
                to_fit=True, batch_size=100, dim=(224, 224),
                n_channels=3, n_frames=30, n_classes=2,
                shuffle=True, seed=42):
        'Initialization'
        self.seed = seed
        np.random.seed(self.seed)
        self.list_IDs = list_IDs
        self.meta = meta
        self.labels = self.meta.label
        self.video_path = Path(videos_foler_path)
        self.video_names = self.meta.index.tolist()
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X1, X2 = self._generate_X(list_IDs_temp)#, X3
        X = [X1,X2]#,X3]
    
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def _generate_X(self, list_IDs_temp):
        'Generates data containing batch_size images'
        # Initialization
        X1 = np.empty((self.batch_size,self.n_frames, *self.dim, self.n_channels),dtype=int)
        X2 = np.empty((self.batch_size,self.n_frames, *self.dim, self.n_channels),dtype=int)
        # X3 = np.empty((self.batch_size,self.n_frames, *self.dim, self.n_channels),dtype=int)
        size = self.dim[0]
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            captures_path = Path(self.video_path / ID)
            temp_1 = np.empty((self.batch_size,self.n_frames,*self.dim,self.n_channels))
            temp_2 = np.empty((self.batch_size,self.n_frames,*self.dim,self.n_channels))
            for j in os.listdir(captures_path/"face_1"):
                im = Image(captures_path/"face_1"/j)
                temp_1[i,h,] = np.asarray(im)
            for k in os.listdir(captures_path/"face_2"):
                im = Image(captures_path/"face_2"/k)
                temp_2[i,k,] =np.asarray(im)

            X = [temp_1,temp_2]#,temp_3]
            np.random.shuffle(X)
            X1[i,] = X[0]
            X2[i,] = X[1]
            # X3[i,] = X[2]

        return X1,X2#,X3
 
    def _generate_y(self, list_IDs_temp):
        'Generates data containing batch_size masks'
        y = np.empty(self.batch_size,dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            lable = self.meta.loc[ID].label
            mask = {"FAKE":1,"REAL":0}
            y[i] = mask.get(lable)

        return y

