from scipy import io
import numpy as np

def load_standard_mat(mat_file, gt=False):

    dataset = io.loadmat(mat_file)
    data = dataset['data'].astype(np.float32)
    if gt:
        labels = dataset['gt'].astype(np.float32)
    else:
        labels = dataset['groundtruth'].astype(np.float32)
    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels

# loading training data with mat format
# Download from http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
def load_segundo_mat(data_path, gt_path, remove_bands=True):
    data = io.loadmat(data_path)['data'].astype(np.float32)
    labels = io.loadmat(gt_path)['groundtruth']

    # Delete several bands for keeping the the same bands with testing dataset (from 224 decrease to 189)
    if remove_bands:
        bands = np.concatenate((np.arange(6, 32),
                                   np.arange(35, 96),
                                   np.arange(97, 106),
                                   np.arange(113, 152),
                                   np.arange(166, 220)))
        data = data[:, :, bands]

    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels


hyperdata_path = './data/Segundo/segundo.mat'
dgt_path = './data/Segundo/groundtruth.mat'

data, groundtruth = load_segundo_mat(hyperdata_path, dgt_path)

path1 = './data/Segundo/sedo.mat'
io.savemat(path1, {'data': data})
path2 = './data/Segundo/gdtruth.mat'
io.savemat(path2, {'groundtruth': groundtruth})