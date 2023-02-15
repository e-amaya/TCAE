import tensorflow as tf
from tensorflow.keras import Model, layers

###### Define model ###########################################
class Encoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.conv_01 = layers.Conv2D(filters = 32, kernel_size=(3,3), strides = 1, padding='same', activation='relu')
        self.conv_02 = layers.Conv2D(filters = 64, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')
        self.conv_03 = layers.Conv2D(filters = 128, kernel_size=(3,3), strides = 1, padding = 'same', activation='relu')
        self.max_pool = layers.MaxPool2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.dense_01 = layers.Dense(units=latent_dim, activation = 'relu')
        
    def call(self, Xt):
        x = self.conv_01(Xt)
        x = self.max_pool(x)
        x = self.conv_02(x)
        x = self.max_pool(x)
        x = self.conv_03(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        zt = self.dense_01(x)
        return zt

class Time_Embedding(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_01 = layers.Dense(32, activation = 'linear')
        self.hidden_02 = layers.Dense(256,activation = 'linear')
        self.hidden_03 = layers.Dense(latent_dim,activation = 'linear')
        
    def call(self, dt):
        x = self.hidden_01(dt)
        x = self.hidden_02(x)
        dtphi = self.hidden_03(x)
        return dtphi

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
    
    def call(self, zt_plus_dtphi):
        x = self.dense_02(zt_plus_dtphi)
        x = self.reshape(x)
        x = self.conv_t_1(x)
        x = self.upsampling(x)
        x = self.conv_t_2(x)
        x = self.upsampling(x)
        x = self.conv_t_3(x)
        x = self.upsampling(x)
        X_t_plus_dt = self.conv_t_4(x)
        return X_t_plus_dt

class TConvAutoEncoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()
        self.time_embedding = Time_Embedding(latent_dim)
        self.addition = tf.keras.layers.Add()
    def call(self, inputs):
        Xt, dt = inputs
        zt = self.encoder(Xt)
        dtphi = self.time_embedding(dt)
        zt_plus_dtphi = self.addition([zt, dtphi])
        X_t_plus_dt = self.decoder(zt_plus_dtphi)
        return X_t_plus_dt

############################################################
