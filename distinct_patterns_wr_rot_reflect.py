import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import augment_image, pattern_number2bin_im

bin_strs_list = list(np.load('bin_str.npy')) # binary strings for valid patterns (single connected area)
selected_patterns = [] # list of patterns where each pattern is not repeated with respect to rotation and reflection
n_transformation = [] # number of different (at the pixel level) transformations per pattern 

while bin_strs_list:
    str_ = bin_strs_list[0]
    pattern_number = int(str_, 2)
    
    # count number of different transformations per pattern
    symmetric_ims = np.unique(augment_image(pattern_number2bin_im(pattern_number)), axis = 0)
    n_transformation.append(len(symmetric_ims))
    
    # only save one transformation of the pattern, discard the rest
    selected_patterns.append(''.join(map(str, symmetric_ims[0].flatten())))
    for p in symmetric_ims:
        p = ''.join(map(str, p.flatten()))
        bin_strs_list.remove(p)
np.save('selected_patterns_no_rot_reflection.npy', selected_patterns)

plt.figure(dpi = 200)
sns.histplot(n_transformation, color = 'k', bins = [1,2,3,4,5,6,7,8,9])
plt.yscale('log')
plt.xlabel('n')
plt.title('Number of distinct symmetries per pattern')
plt.show()