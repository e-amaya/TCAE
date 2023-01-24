import os
import numpy as np
import imageio
from IPython.display import Image

def pattern_number2bin_im(pattern):
    """
    Takes an integer in [0-65535], converts it to its binary representation, and reshapes it as a 4x4 binary array
    
    Args:
    pattern (int): integer in [0-65535]
    """
    string_repr = np.base_repr(pattern).zfill(16)
    return np.fromstring(' '.join(string_repr), dtype=int, sep=' ').reshape((4,4))

def bin_im2pattern_number(bin_im):
    """
    Takes an n by m binary numpy array, converts it to a binary string, outputs the decimal representation of the binary number
    
    Args:
    bin_im (numpy array): binary array with shape (n,m) and dtype = 'int'
    """
    string_rep = ''.join(map(str, bin_im.flatten()))
    return int(string_rep, base = 2)

def augment_bin_im(bin_im):
    """
    Takes an n by n binary numpy array and outputs a list of its dihedral group (d4) transformations
    
    Args:
    bin_im (numpy array): binary array with shape (n,n) and dtype = 'int'
    """
    transformed_data = [bin_im] # identity
    for i in range(1,3+1):  
        transformed_data.append(np.rot90(bin_im, k=i, axes=(0, 1))) # rotations
    # Reflections
    transformed_data.append(np.flip(bin_im,axis=0))
    transformed_data.append(np.flip(bin_im,axis=1))
    transformed_data.append(np.transpose(bin_im, axes=(1,0)))
    transformed_data.append(np.transpose(bin_im[::-1,::-1], axes=(1,0)))
    return transformed_data

def matplotlib_fig_to_np_array(fig):
    """
    Takes a matplotlib figure object, and converts it to a numpy array
    
    Args:
    fig: matplotlib figure object
    """
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def embed_images(image_list, duration = 0.04):
    """
    Takes a list of images (each image is a numpy array), and embeds a gif containing the images in the jupyter notebook    
    Args:
    image_list (list): list of numpy arrays, each containing an image
    duration (positive float): time duration of the gif embedded
    """
    temp_file = 'temporary_file_[$.Y$n[Y`U*)${+S.gif'
    imageio.mimsave(temp_file, image_list, format='GIF' ,duration = duration)
    display(Image(data=open(temp_file,'rb').read(), format='png'))
    os.remove(temp_file)