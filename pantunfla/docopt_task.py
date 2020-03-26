"""
Usage: pantunfla.docopt_task 

Options:
    -h --help     Show this screen.

"""
from docopt import docopt

import pantunfla.Model_jc as mdl # Your model.py file.
import pantunfla.download_and_pre_process_for_virtual_machine as dppvm
import pantunfla.pre_process_funcs as ppf
from pathlib import Path
import subprocess
import pandas as pd
import os
import tensorflow as tf
from keras.applications import ResNet152V2, ResNet50
import time
import multiprocessing
from keras.utils import multi_gpu_model
import gc
gc.enable()

print("0.0")
if __name__ == '__main__':
    print("0.1")
    try:
        arguments = docopt(__doc__)
    except:
        print("la neta no valedor")
    print("1")
    print("2")
    print("3")
    dppvm.DEST = Path("destination_directory")
    dims = (224,224)
    channels = 3
    n_frames = 30
    ppf.frames_per_video = n_frames
    print("4")
    saved_model_path = "weights-improvement-{epoch:02d}.hdf5"
    print("5")

    DATA = Path()
    DEST = dppvm.DEST
    print("6")
    print("7")
    print("8")
    print("9")
    path_video_files = dppvm.DEST/'videos'
    path_meta = DEST/'metadata'/'all_meta.json'
    all_meta = pd.read_json(path_meta)
    print("10")

    
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor="val_acc",verbose=1,save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor= "val_acc", min_delta = 0.01, patience = 5, restore_best_weights=True)
    callbacks_list = [earlystop]#,checkpoint
    optimizer = tf.keras.optimizers.Adam()
    binloss = tf.keras.losses.BinaryCrossentropy()
    acc = tf.keras.metrics.Accuracy()

    val_msk = int(len(all_meta) * 0.9)
    ppf.DataGenerator()
    val   = ppf.DataGenerator(all_meta[val_msk:].index,videos_folder_path=DEST/"captures",meta=all_meta[val_msk:])
    gener = ppf.DataGenerator(all_meta[:val_msk].index,videos_folder_path=DEST/"captures",meta=all_meta[:val_msk])
    # gener = ppf.DataGenerator(all_meta.index,video_path=all_meta.path,meta=all_meta)
    model =  model = mdl.make_model(n_frames,dims,channels)
    # model = tf.keras.utils.multi_gpu_model(model,2)
    model.compile(optimizer= optimizer, loss = binloss, metrics = [acc])

    model.summary()
    print("11")
    model.fit_generator(generator = gener,callbacks=callbacks_list,validation_data=val,use_multiprocessing=True,workers=100,verbose=1,epochs=500,max_queue_size=50)

    # Make_predicctions
