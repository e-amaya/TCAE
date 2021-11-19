import os 
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
from helper_functions import *
from TCAE import *

experiment_name = "next_frame_pred_no_arms_01"
epochs = 100

# Create Directories to save results ##########################

base_dir = os.path.join(os.getcwd(),experiment_name)
weights_dir = os.path.join(base_dir,'weights')
history_dir = os.path.join(base_dir,'history')

create_dir(base_dir)
create_dir(weights_dir)
create_dir(history_dir)

###### Load data ##############################################

file_list = glob.glob("/home/ubuntu/am-cvae/data/112x112_pulse/*")
file_list_training = [file for file in file_list if not "rec600" in file ] # Hold-out rec600 video for val
file_list_val = [file for file in file_list if "rec600" in file ]

start_frame = 30
end_frame = 200
file_shape = [360, 112, 112]

x_train = load_list_files_as_array(
    file_list_training, 
    file_shape,
    start_index = start_frame, 
    end_index = end_frame
)
x_val = load_list_files_as_array(
    file_list_val, 
    file_shape, 
    start_index = start_frame, 
    end_index = end_frame
)

x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)

video_length = abs(start_frame - end_frame)
x_train,y_train,dt_train = timeshift_dataset(x_train, n_frame=20, video_length= video_length)
x_val,y_val,dt_val = timeshift_dataset(x_val, n_frame=20, video_length= video_length)



latent_dim = 2048
tcae = TConvAutoEncoder(latent_dim)
tcae.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
)

filename_history = os.path.join(history_dir,f'tcvae_time_embedding_epochs_100_log.csv')
history_logger=tf.keras.callbacks.CSVLogger(filename_history, separator=",", append=True)

history = tcae.fit(
    (x_train,dt_train),
    y_train,
    batch_size = 128,
    epochs = epochs,
    shuffle = True,
    validation_data = ((x_val,dt_val),y_val),
    callbacks=[history_logger]
)

tcae.save_weights(os.path.join(weights_dir,f'tcae_epoch_{epochs}_model.h5'))
