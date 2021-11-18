import os
import numpy as np
import tqdm

def create_dir(path):
    '''
    Creates a directory from a given path if that directory does not exist.
    
    Args:
    path (str): Location where the directory is going to be created
    '''
    if not os.path.exists(path):
        os.makedirs(path)  
        
def load_list_files_as_array(file_list, file_shape, start_index = 30, end_index = 200):
    '''
    Creates an array with size = (total_number of samples, height, width)
    and then populates it using a list of paths for .npy files.
    
    Args:
    file_list (list): List of paths (str) for each npy file
    file_shape (list): shape of numpy array of npy file
    start_index (int): start index for slicing npy file
    end_index (int): end index for slicing npy file
    '''
    file_shape[0] = end_index - start_index
    n_files = len(file_list) # number of npy files
    n_samples_file = file_shape[0] # number of samples per file
    file_shape[0] *= n_files # total number of samples
    result = np.zeros((file_shape), dtype = 'float32')
    for i, file in enumerate(tqdm.tqdm(file_list)):
        array = np.load(file)
        array = array[start_index:end_index]
        result[n_samples_file*(i):n_samples_file*(i+1)] = array
    return result


def temporal_dataset(data, n_frame = 20, video_length=170):
    '''
    Given, an Image from a video at time t: "Xt" and the number of timesteps we want to predict: "n_frame" 
    this function produces (x,y,dt) where:
    x =  [Xt,   Xt,   Xt,....,   Xt] "Xt" is repeated "n_frame" times.
    y =  [Xt+0, Xt+1, Xt+2,...., Xt+n_frame] selection of images in video from time "t" to time "t+n_frame".
    dt = [0,     1, ....,    n_frame] time difference values between "x" and "y".
    The ouput is a concatenation of (x,y,dt) for all different Xt(from t = 0 to t = "video_length" - "n_frame"), 
    for all videos in "data".
    
    Args:
    data (numpy array): numpy array of concatenated videos.
    n_frame (int): number of frames we want to predict.
    video_length (int): length of each concatenated video in data.
    '''
    x_indices = tuple(slice(i*video_length,(i+1)* video_length - n_frame,1) for i in range(data.shape[0]// video_length))
    y_indices = tuple(slice(i*video_length + j,i*video_length + n_frame + j,1) for i in range(data.shape[0]//video_length) for j in range(video_length - n_frame))
    x = np.repeat(data[np.r_[x_indices]], n_frame, axis=0)
    y = data[np.r_[y_indices]]
    dt = np.tile(np.r_[slice(0,n_frame,1)],(data.shape[0]//video_length)* (video_length - n_frame))
    dt = dt.reshape((dt.shape[0], 1)).astype('float32')
    return (x,y,dt)