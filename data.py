from __future__ import division, print_function
from collections import Counter
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image


LABEL_FILE = 'data/trainLabels.csv'

RANDOM_STATE = 9
FEATURE_DIR = 'data/features'

# channel standard deviations
STD = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32)

# channel means
MEAN = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32)

# set of resampling weights that yields balanced classes
BALANCE_WEIGHTS = np.array([1.3609453700116234,  14.378223495702006, 
                            6.637566137566138, 40.235967926689575, 
                            49.612994350282484])

# for color augmentation, computed with make_pca.py
U = np.array([[-0.56543481, 0.71983482, 0.40240142],
              [-0.5989477, -0.02304967, -0.80036049],
              [-0.56694071, -0.6935729, 0.44423429]] ,dtype=np.float32)
EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

def get_labels(names, labels=None, per_patient=False):
    
    if labels is None:
        labels = pd.read_csv(LABEL_FILE, 
                             index_col=0).loc[names].values.flatten()

    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels


def get_image_files(datadir, left_only=False):
    fs = glob('{}/*'.format(datadir))
    if left_only:
        fs = [f for f in fs if 'left' in f]
    return np.array(sorted(fs))


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]
