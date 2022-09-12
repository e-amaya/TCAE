# ConvAE with time embedding
# Data augmentation in training data: Rotation

import os 
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
import datetime

experiment_name = "next_frame_pred_no_arms_02"

# Create Directories to save results

base_dir = os.path.join(os.getcwd(),experiment_name)
weights_dir = os.path.join(base_dir,'weights')
history_dir = os.path.join(base_dir,'history')
images_dir = os.path.join(base_dir,'images')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  

create_dir(base_dir)
create_dir(weights_dir)
create_dir(history_dir)
create_dir(images_dir)

###### Define model ###########################################

class Time_Embedding(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_01 = layers.Dense(32, activation = 'linear')
        self.hidden_02 = layers.Dense(256,activation = 'linear')
        self.hidden_03 = layers.Dense(2048,activation = 'linear')
        
    def call(self, scalar):
        x = self.hidden_01(scalar)
        x = self.hidden_02(x)
        x = self.hidden_03(x)
        return x

class Encoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.conv_01 = layers.Conv2D(filters = 32, kernel_size=(3,3), strides = 1, padding='same', activation='relu')
        self.conv_02 = layers.Conv2D(filters = 64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')
        self.conv_03 = layers.Conv2D(filters = 128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')
        self.max_pool = layers.MaxPool2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.dense_01 = layers.Dense(units=latent_dim)
        
    def call(self, input_tensor):
        x = self.conv_01(input_tensor)
        x = self.max_pool(x)
        x = self.conv_02(x)
        x = self.max_pool(x)
        x = self.conv_03(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense_01(x)
        return x

class Decoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_02 = layers.Dense(units=14*14*128, activation='relu')
        self.reshape = layers.Reshape(target_shape=(14,14,128))
        self.conv_t_1 = layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv_t_2 = layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv_t_3 = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv_t_4 = layers.Conv2DTranspose(filters=1, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
        self.upsampling = layers.UpSampling2D(size=(2,2))
    
    def call(self, encodings):
        x = self.dense_02(encodings)
        x = self.reshape(x)
        x = self.conv_t_1(x)
        x = self.upsampling(x)
        x = self.conv_t_2(x)
        x = self.upsampling(x)
        x = self.conv_t_3(x)
        x = self.upsampling(x)
        x = self.conv_t_4(x)
        return x

class ConvAutoEncoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()
        self.time_embedding = Time_Embedding()
        self.addition = tf.keras.layers.Add()
    def call(self, inputs):
        input_tensor, scalar = inputs
        t = self.time_embedding(scalar)
        x = self.encoder(input_tensor)
        x = self.addition([x,t])
        x = self.decoder(x)
        return x

############################################################

# # Loading data

file_list = glob.glob("/home/ubuntu/am-cvae/data/112x112/*")
file_list = [i for i in file_list if "pulse" in i ]
file_list.sort()

start_frame = 30
end_frame = 200

data = np.load(file_list[0])[start_frame:end_frame]
for file in file_list[1:]:
    if "rec600" not in file:
        data = np.concatenate([data,np.load(file)[start_frame:end_frame]])
data = np.expand_dims(data, -1).astype("float32")

val_data = np.load([file for file in file_list if "rec600" in file][0])[start_frame:end_frame]
val_data = np.expand_dims(val_data, -1).astype("float32")

print("original data size",data.shape)

def augment_data(data, transformation="rotation"):
    assert data.ndim == 4, 'array should have dims (samples,height,width,channels)'
    
    transformed_data = {}
    
    if transformation == 'rotation' or transformation == 'both':
        transformed_data['rotation'] = np.zeros([j*3 if i == 0 else j for i,j in enumerate(data.shape)])
        for i in range(1,3+1):  
            transformed_data['rotation'][data.shape[0]*(i-1):data.shape[0]*i] = np.rot90(data, k=i, axes=(1, 2))
    
    if transformation == 'reflection' or transformation == 'both':    
        transformed_data['reflection'] = np.zeros([j*4 if i == 0 else j for i,j in enumerate(data.shape)])
        transformed_data['reflection'][data.shape[0]*0:data.shape[0]*1] = np.flip(data,axis=1)
        transformed_data['reflection'][data.shape[0]*1:data.shape[0]*2] = np.flip(data,axis=2)
        transformed_data['reflection'][data.shape[0]*2:data.shape[0]*3] = np.transpose(data, axes=(0,2,1,3))
        transformed_data['reflection'][data.shape[0]*3:data.shape[0]*4] = np.transpose(data[:,::-1,::-1,:], axes=(0,2,1,3))
    
    return transformed_data

def temporal_dataset(data, n_frame = 20, video_length=170):
    x_indices = tuple(slice(i*video_length,(i+1)* video_length - n_frame,1) for i in range(data.shape[0]// video_length))
    y_indices = tuple(slice(i*video_length + j,i*video_length + n_frame + j,1) for i in range(data.shape[0]//video_length) for j in range(video_length - n_frame))
    x = np.repeat(data[np.r_[x_indices]], n_frame, axis=0)
    y = data[np.r_[y_indices]]
    dt = np.tile(np.r_[slice(0,n_frame,1)],(data.shape[0]//video_length)* (video_length - n_frame))
    return (x,y,dt)



data = np.concatenate([data, augment_data(data, transformation="rotation")['rotation']])
print("data plus augmentation size", data.shape)
video_length = abs(start_frame - end_frame)
x_train,y_train,dt_train = temporal_dataset(data, n_frame=20, video_length= video_length)
x_test,y_test,dt_test = temporal_dataset(val_data, n_frame=20, video_length= video_length)

print("x_train size: ", x_train.shape)
print("y_train size: ", y_train.shape)
print("dy_train size: ", dt_train.shape)

print("x_test size: ", x_test.shape)
print("y_test size: ", y_test.shape)
print("dy_test size: ", dt_test.shape)

conv_ae = ConvAutoEncoder(2048)
conv_ae.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
)

filename = os.path.join(history_dir,f'cvae_time_embedding_epochs_100_log.csv')
history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

mc_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(weights_dir,'cvae_t_emb{epoch:08d}.h5'),
        save_weights_only = True,
        period = 10,
        )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = conv_ae.fit(
    (x_train,dt_train),
    y_train,
    batch_size = 128,
    epochs = 100,
    shuffle = True,
    validation_data = ((x_test,dt_test),y_test),
    callbacks=[
        history_logger,
        mc_callback,
        tb_callback,
        ],
#    verbose = 2,
)

conv_ae.save_weights(os.path.join(weights_dir,f'cvae_time_emb_epochs_100_model.h5'))

