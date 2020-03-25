import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, LSTM, Dropout, AveragePooling2D
from keras.layers import Flatten, BatchNormalization, Convolution2D,Input
from keras. layers import TimeDistributed, Reshape, concatenate
from keras.layers import Activation, MaxPool2D
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam
import pantunfla.pre_process_funcs as funcs
import tensorflow as tf
from keras.applications import ResNet152V2, ResNet50
tf.compat.v1.disable_eager_execution()
# Setting tf for my gpu
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import metadata
# meta_path=r"metadata.json"
# df = meta = pd.read_json(meta_path).T

# # # # # Videos # # # # #
# test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
# # test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
# path_to_videos = r"D:\Carpetas en escritorio\Kaggle\Deepfake detection\Data\deepfake-detection-challenge\train_sample_videos"
# meta["path"] = path_to_videos + r'/' + meta.index

# X = funcs.DataGenerator(meta.index,video_path=meta.path,meta=meta)

# Set input shape
# dims = (224, 224)
# channels = 3
# n_frames = 30


# # # # # # # Create Model # # # # # #
# Create inputs
# def make_model(n_frames,dims,channels):
#     input_1 = tf.keras.layers.Input(shape=[ n_frames, *dims, channels], name="input_1") 
#     input_2 = tf.keras.layers.Input(shape=[ n_frames, *dims, channels], name="input_2")
#     # input_3 = Input(shape=[ n_frames, *dims, channels], name="input_3")
#     # Create first part of the model
#     # x1 = TimeDistributed(ResNet50())(input_1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16,kernel_size=(7,7)),name="1.1")(input_1)
#     x1 = tf.keras.layers.Activation("relu", name="act1.1")(x1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32,kernel_size=(5,5)),name="1.2")(x1)
#     x1 = tf.keras.layers.Activation("relu", name="act1.2")(x1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="1.2b")(x1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64,kernel_size=(3,3)),name="1.3")(x1)
#     x1 = tf.keras.layers.Activation("relu", name="act1.3")(x1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="1.3b")(x1)
#     # x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(),name="1.3m")(x1)
#     x1 = tf.keras.layers.MaxPool3D()(x1)
#     x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x1)
#     # x1 = tf.keras.layers.LSTM(2,name="1.lstm")(x1)
#     # x1 = tf.keras.layers.BatchNormalization(name="1.lstmb")(x1)
#     # mod_1 = tf.keras.models.Model(inputs=input_1, outputs = x1)
#     # Create second part of model
#     # x2 = TimeDistributed(ResNet50())(input_2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16,kernel_size=(7,7)),name="2.1")(input_2)
#     x2 = tf.keras.layers.Activation("relu", name="act2.1")(x2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32,kernel_size=(5,5)),name="2.2")(x2)
#     x2 = tf.keras.layers.Activation("relu", name="act2.2")(x2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="2.2b")(x2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64,kernel_size=(3,3)),name="2.3")(x2)
#     x2 = tf.keras.layers.Activation("relu", name="act2.3")(x2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="2.3b")(x2)
#     # x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(),name="2.3m")(x2)
#     x2 = tf.keras.layers.MaxPool3D()(x2)
#     x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x2)
#     # mod_2 = tf.keras.models.Model(inputs=input_2, outputs = x2)
#     # x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Concatenate([mod_1.output, mod_2.output]))
#     x4 = tf.keras.layers.concatenate(inputs=[x1,x2],axis =-1)
#     x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
#     x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
#     # x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
#     # x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
#     # x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
#     x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation = "sigmoid"))(x4)
#     x4 = tf.keras.layers.MaxPool1D()(x4)
#     x4 = tf.keras.layers.LSTM(1,name="2.lstm")(x4)
#     x4 = tf.keras.layers.Dense(1,activation="sigmoid")(x4)
#     # x2 = tf.keras.layers.LSTM(2,name="2.lstm")(x2)
#     # x2 = tf.keras.layers.BatchNormalization(name="2.lstmb")(x2)
#     # mod_2 = tf.keras.models.Model(inputs=input_2, outputs = x2)
#     # Create third part of model
#     # x3 = TimeDistributed(ResNet50())(input_3)
#     # x3 = TimeDistributed(Conv2D(32,kernel_size=(7,7)),name="3.1")(input_3)
#     # x3 = Activation("relu", name="act3.1")(x3)
#     # x3 = TimeDistributed(Conv2D(64,kernel_size=(5,5)),name="3.2")(x3)
#     # x3 = Activation("relu", name="act3.2")(x3)
#     # x3 = TimeDistributed(BatchNormalization(),name="3.2b")(x3)
#     # x3 = TimeDistributed(Conv2D(128,kernel_size=(3,3)),name="3.3")(x3)
#     # x3 = Activation("relu", name="act3.3")(x3)
#     # x3 = TimeDistributed(BatchNormalization(),name="3.3b")(x3)
#     # x3 = TimeDistributed(MaxPool2D(),name="3.3m")(x3)
#     # x3 = TimeDistributed(Flatten())(x3)
#     # x3 = LSTM(2,name="3.lstm")(x3)
#     # x3 = BatchNormalization(name="3.lstmb")(x3)
#     # mod_3 = Model(inputs=input_3, outputs = x3)
#     # Join parts 1,2,3 of model
#     # x4 =  tf.keras.layers.concatenate([mod_1.output, mod_2.output])#, mod_3.output
#     # x4 = tf.keras.layers.Dense(24, activation ="relu")(x4)
#     # x4 = tf.keras.layers.Dense(24, activation ="relu")(x4)
#     # x4 = tf.keras.layers.Dense(24, activation ="relu")(x4)
#     # x4 = tf.keras.layers.Dense(24, activation ="relu")(x4)
#     # x4 = Dense(24, activation ="relu")(x4)
#     # x4 = Dense(24, activation ="relu")(x4)
#     # x4 = tf.keras.layers.Dense(1,  activation ="sigmoid")(x4)
#     # mod_4 = tf.keras.models.Model(inputs=[mod_1.input,mod_2.input], outputs=x4)#,mod_3.input
#     mod_4 = tf.keras.models.Model(inputs=[input_1,input_2], outputs=x4)
#     return mod_4

import tensorflow as tf
def make_model(n_frames,dims,channels):
    input_1 = tf.keras.layers.Input(shape=[ n_frames, *dims, channels], name="input_1") 
    input_2 = tf.keras.layers.Input(shape=[ n_frames, *dims, channels], name="input_2")

    x1 = tf.keras.layers.ConvLSTM2D(32,kernel_size=(7,7),name="1.1", return_sequences=True)(input_1)
    x1 = tf.keras.layers.ConvLSTM2D(64,kernel_size=(5,5),name="1.2")(x1)
    x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="1.2b")(x1)
    x1 = tf.keras.layers.Conv2D(128,kernel_size=(3,3),name="1.3")(x1)
    x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="1.3b")(x1)
    x1 = tf.keras.layers.MaxPool2D()(x1)
    # x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x1)

    x2 = tf.keras.layers.ConvLSTM2D(32,kernel_size=(7,7),name="2.1",return_sequences=True)(input_2)
    x2 = tf.keras.layers.ConvLSTM2D(64,kernel_size=(5,5),name="2.2")(x2)
    x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="2.2b")(x2)
    x2 = tf.keras.layers.Conv2D(128,kernel_size=(3,3),name="2.3")(x2)
    x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(),name="2.3b")(x2)
    x2 = tf.keras.layers.MaxPool2D()(x2)
    # x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x2)

    x4 = tf.keras.layers.concatenate(inputs=[x1,x2],axis =-1)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24,activation = "relu"))(x4)
    x4 = tf.keras.layers.MaxPool2D()(x4)
    # x4 = tf.keras.layers.LSTM(1,name="lstm")(x4)
    x4 = tf.keras.layers.Flatten()(x4)
    x4 = tf.keras.layers.Dense(1,activation="sigmoid")(x4)

    mod_4 = tf.keras.models.Model(inputs=[input_1,input_2], outputs=x4)
    return mod_4





# make_model(30,(224,224),3).summary()
















# # Create callbacks, metrics, loss, and Generator # #
# Callbacks
# saved_model_path = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(saved_model_path, monitor="val_accuracy",verbose=1,save_best_only=True)
# earlystop = EarlyStopping(monitor= "val_accuracy", min_delta = 0.01, patience = 5, restore_best_weights=True)
# callbacks_list = [checkpoint, earlystop]

# Optimizer
# optimizer = Adam()

# Loss
# binloss = BinaryCrossentropy()

# Metrics 
# acc = Accuracy()

# Generator
# gener = funcs.DataGenerator(meta[:300].index,video_path=meta[:300].path,meta=meta[:300])
# val = funcs.DataGenerator(meta[300:].index,video_path=meta[300:],meta=meta[300:])
# Compile and set for training
# mod_4.compile(optimizer= optimizer, loss = binloss, metrics = [acc])


