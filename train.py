# -*- coding: utf-8 -*-
import numpy as np

import nn
import data

def main():

    net = nn.create_net()

    train_files = data.get_image_files('croppedData')
    train_names = data.get_names(train_files)
    train_labels = data.get_labels(train_names).astype(np.float32)
    net.fine_tune(train_files, train_labels)

    test_files = data.get_image_files('croppedData')
    test_names = data.get_names(test_files)
    test_labels = data.get_labels(test_names).astype(np.float32)
    print("test labels:")
    print(test_labels)
    net.my_test(test_files, test_labels) #my_test根本就不存在是smg？ make_submission?

if __name__ =='__main__':
    main()

'''
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
'''
