# For each simulated light pattern,
# the following code finds min_{ʎ,t_x,t_y}[MSE(S_θ(x),y)] using GD,
# where: x = image of microtubule network at t = 0
#        y = image of microtubule networl at t = 169 
#       S_θ = affine transformation specified by θ = [ [ʎ, 0, t_x],
#                                                      [0, ʎ, t_y] ]


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU faster than GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from helper_functions import pattern_number2dir, dir2pattern_number, write_csv_append_dict_as_row
from affine_transformation_gradient_descent import *


def process_files_for_spatial_transformer(path):
    threshold = .05
    x = load_img(path, color_mode = "grayscale")
    x = img_to_array(x) / 255.
    x = (x > threshold).astype('float32')
    x = np.expand_dims(x, 0)
    x = tf.constant(x)    
    return x

def run_affine_transform_GD_using_opt_params(dir_list, csv_file, opt_params, max_steps = 1000, patience = 20, epsilon = 1e-5):
    headersCSV = ['id','optimizer','lr', 'loss_array','final_loss', 'theta']   
    pd.DataFrame(columns = headersCSV).to_csv(csv_file, index = False)
    for dir_ in tqdm.tqdm(dir_list):
        x = process_files_for_spatial_transformer(os.path.join(dir_, 'frame_0000.png'))
        y = process_files_for_spatial_transformer(os.path.join(dir_, 'frame_0169.png'))
        for opt_name, lr_list in opt_params.items():
            for lr in lr_list:
                output_dict = affine_transformation_gradient_descent(
                    x,
                    y,
                    id_ = dir2pattern_number(dir_),
                    opt_name = opt_name, 
                    lr = lr, 
                    max_steps = max_steps, 
                    patience = patience, 
                    epsilon = epsilon,
                    save_theta_array = False,
                )
                write_csv_append_dict_as_row(csv_file, headersCSV,  output_dict)


data_dir = os.path.join(os.getcwd(), 'sim_data_112x112/') 
selected_patterns = [int(str_, 2) for str_ in np.load('selected_patterns_no_rot_reflection.npy')]
selected_patterns_dirs = [pattern_number2dir(pattern, data_dir) for pattern in selected_patterns]

csv_file = 'gd_affine_transformation_loss_min.csv'
opt_params = {'Adam': [1e-2, 1e-3], 'SGD': [1e-07]}
run_affine_transform_GD_using_opt_params(
    selected_patterns_dirs, 
    csv_file, 
    opt_params,
    max_steps = 2000, 
    patience = 20, 
    epsilon = 1e-5,
)