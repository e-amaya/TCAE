# Image Processing Functions

from skimage import filters
from skimage.morphology import remove_small_objects
from skimage.exposure import rescale_intensity
from scipy.ndimage.morphology import binary_fill_holes
import imageio
import glob
import tqdm
import numpy as np

def thresholding(images, method = 'triangle'):
    '''
    Segments images in numpy array by applying a given thresholding method. 

    Args:
    images (numpy array): numpy array with images we want to segment. shape = (n, height, width, channels)
    method (str): thresholding method. Available options can be found in:
                  https://scikit-image.org/docs/stable/api/skimage.filters.html
    '''
    assert images.ndim == 4, "shape of 'images' must be = (n, image_h, image_w, channels)"
    result = np.empty(images.shape, dtype = 'bool')
    method = 'threshold_'+ method
    method_to_call = getattr(filters, method)
    for i in range(images.shape[0]):
        im = rescale_intensity(images[i,:,:], out_range=(0, 1))
        threshold = method_to_call(im)
        im = im > threshold
        im = binary_fill_holes(im)
        result[i,:,:,:] = im
    return result

def remove_objects(bin_images, min_size=64):
    '''
    Remove objects smaller than 'min_size' in array of binary masks. 

    Args:
    bin_images (numpy array): numpy array with binary images. shape = (n, height, width, channels)
    min_size (int): objects smaller than 'min_size will be removed.
    '''
    assert bin_images.ndim == 4, "shape of image must be = (n, image_h, image_w, channels)"
    result = np.empty(bin_images.shape, dtype = 'bool')
    for i in range(bin_images.shape[0]):
        result[i,:,:,:] = remove_small_objects(bin_images[i,:,:,:], min_size = min_size)
    return result

def count_pixels(bin_images):
    '''
    Counts the number of pixels with 'True' value per image in array of binary images.
    
    Args:
    bin_images (numpy array): numpy array with binary images. shape = (n, height, width, channels)
    '''
    assert bin_images.ndim == 4, "shape of image must be = (n, image_h, image_w, channels)"
    return np.sum(bin_images[:,:,:,0], axis=(1,2))

def create_gif(dst_path, images_dir):
    '''
    Creates a GIF of PNG images saved in a directory
    
    Args:
    dst_path (str): Destination Path
    images_dir (str): Source Path with PNG images
    '''
    with imageio.get_writer(dst_path, mode='I') as writer:
        filenames = glob.glob(images_dir + '/*')
        filenames.sort()
        for filename in tqdm.tqdm(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
