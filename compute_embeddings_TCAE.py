import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import tqdm
import numpy as np
import tensorflow as tf
from TCAE import TConvAutoEncoder

experiment_name = "prediction_am_sim_002"
base_dir = os.path.join(os.getcwd(), experiment_name)
weights_dir = os.path.join(base_dir,'weights')
data_dir = os.path.join(os.getcwd(), 'sim_data_112x112/') 
dir_list = sorted(glob.glob(os.path.join(data_dir, '*')))

N_STEPS = 22
VIDS_PER_STEP = 523  # 11506/523 = 22
N_BATCHES = 34 
BATCH_SIZE = 2615  # (523*170)/2615 = 34

# Instatiate model
tcae = TConvAutoEncoder(64)
# dummy data to "connect" the model needed before loading weights
tcae((np.zeros((1,112,112,1)), np.zeros((1,1))))
tcae.load_weights(weights_dir + f'/TCAE_epoch_{29}.h5')


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels = 1)
    img = tf.image.convert_image_dtype(img, tf.float32) 
    return img

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

embeddings_64d = np.zeros((11506*170, 64), dtype = 'float32')

for step in tqdm.tqdm(range(N_STEPS)):
    paths = []
    for dir_ in dir_list[VIDS_PER_STEP * step: VIDS_PER_STEP * (step+1)]:
        paths += sorted(glob.glob(dir_ + '/*'))
    data_set = tf.data.Dataset.list_files(paths, shuffle=False)
    data_set = data_set.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_set = configure_for_performance(data_set)  
    for i, batch in enumerate(data_set):
        embeddings_64d[
            BATCH_SIZE*(N_BATCHES*step + i) : BATCH_SIZE*(N_BATCHES*step + i+1)
        ]  =  tcae.encoder(batch)
        
np.save(base_dir + '/embedding64d.npy', embeddings_64d.reshape(11506,170,64))