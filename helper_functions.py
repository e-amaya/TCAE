import os
import numpy as np
import imageio
from IPython.display import Image
from csv import DictWriter

def pattern_number2dir(pattern, data_dir):
    """
    Given an integer in the range of [0-65535] which represents a pattern number,
    the function outputs the path of the directory where the video data for that 
    specific pattern is stored.
    
    Args:
    pattern (int): integer in [0-65535]
    data_dir (str): parent directory of all data
    """
    prefix = 'pattern_'
    pattern = str(pattern).zfill(5) 
    return os.path.join(data_dir,f'{prefix}{pattern}')

def dir2pattern_number(dir_):
    """
    The function retrieves the pattern number from the directory where its data is stored
    
    Args:
    dir_ (str): the location of the directory where the video data for a specified pattern is stored.
    """
    d = dir_
    prefix = 'pattern_'
    return int(d[d.find(prefix) + len(prefix):])

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

def augment_image(image):
    """
    Takes an n by n numpy array and outputs a list of its dihedral group (d4) transformations
    
    Args:
    image (numpy array): array with shape (n,n)
    """
    transformed_data = [image] # identity
    for i in range(1,3+1):  
        transformed_data.append(np.rot90(image, k=i, axes=(0, 1))) # rotations
    # Reflections
    transformed_data.append(np.flip(image,axis=0))
    transformed_data.append(np.flip(image,axis=1))
    transformed_data.append(np.transpose(image, axes=(1,0)))
    transformed_data.append(np.transpose(image[::-1,::-1], axes=(1,0)))
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
    
def write_csv_append_dict_as_row(csv_file, headersCSV, dict_):
    '''
    Adds a dictionary as a row to an existing CSV file"
    Args:
    csv_file(str): path of CSV file that will be modified
    headersCSV(list): A list of strings representing the names of the columns in the CSV file
    dict_(dict): Dictionary to be added as a row
    '''
    # First, open the old CSV file in append mode, hence mentioned as 'a'
    # Then, for the CSV file, create a file object
    with open(csv_file, 'a', newline='') as f_object:
        # Pass the CSV  file object to the Dictwriter() function
        # Result - a DictWriter object
        dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
        # Pass the data in the dictionary as an argument into the writerow() function
        dictwriter_object.writerow(dict_)
        # Close the file object
        f_object.close()