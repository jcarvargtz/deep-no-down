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
from tqdm import tqdm
import keras
from PIL import Image
gc.enable()

print("0.0")
if __name__ == '__main__':

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    print("0.1")
    try:
        arguments = docopt(__doc__)
    except:
        print("la neta no valedor")
    dppvm.DEST = Path("destination_directory")
    dims = (224,224)
    channels = 3
    n_frames = 30
    ppf.frames_per_video = n_frames
    saved_model_path = "weights-improvement-{epoch:02d}.hdf5"

    DATA = Path()
    DEST = dppvm.DEST

    path_video_files = dppvm.DEST/'videos'
    path_meta = DEST/'metadata'/'all_meta.json'
    meta = pd.read_json(path_meta)
    all_meta = meta[meta["good"]].copy()



    # # bla bla mascara para ver ver del df, cuales son los videos que se procesaron    
    # all_meta["good"] = False
    # for folder in tqdm(os.listdir(DEST/"captures")):
    #     p1 = DEST/"captures"/folder/"face_1"
    #     p2 = DEST/"captures"/folder/"face_2"
    #     if p1.exists() and p2.exists():
    #         for photo_1 in os.listdir(p1):
    #             try: 
    #                 im_1 = Image.open(p1/photo_1)
    #             except: 
    #                 os.remove(p1/photo_1)
    #         for photo_2 in os.listdir(p2):
    #             try:
    #                 im_2 = Image.open(p2/photo_2)
    #             except:
    #                 os.remove(p2/photo_2)
    #         if len(os.listdir(p1)) == 30 and len(os.listdir(p1)) == 30:
    #             all_meta.loc[folder,"good"] = True
    print("oh yuuuur")
    all_meta.to_json(path_meta)
    print("finish")
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor="val_acc",verbose=1,save_best_only=True)
    
    
    # # # # # # # # # # # # uncomment to train
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    print("1")    
    earlystop = keras.callbacks.EarlyStopping(monitor= "val_acc", min_delta = 0.01, patience = 5, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(saved_model_path, monitor="val_acc",verbose=1,save_best_only=True)
    callbacks_list = [earlystop, checkpoint]#,checkpoint

    print("2")
    optimizer = tf.keras.optimizers.SGD()
    binloss = keras.losses.BinaryCrossentropy()
    acc = keras.metrics.Accuracy()

    print("3")
    val_msk = int(len(all_meta) * 0.9)
    val   = ppf.DataGenerator(list(all_meta[val_msk:].index), DEST/"captures", meta=all_meta[val_msk:])
    gener = ppf.DataGenerator(list(all_meta[:val_msk].index), DEST/"captures", meta=all_meta[:val_msk])

    print("4")
    model = mdl.make_model(n_frames,dims,channels)

    print("5")
    # model = tf.keras.utils.multi_gpu_model(model,2)

    print("6")
    print("7")
    print("8")
    print("9")

    print("10")

    # gener = ppf.DataGenerator(all_meta.index,video_path=all_meta.path,meta=all_meta)
    model.compile(optimizer= optimizer, loss = binloss, metrics = [acc])

    
    print("11")
    print("Ahí les va!")
    model.fit_generator(generator = gener,callbacks=callbacks_list,validation_data=val,verbose=1,epochs=500, workers=0)#, workers = 4)#, use_multiprocessing=True,workers=100,max_queue_size=50)

    # # Make_predicctions


