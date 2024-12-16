import numpy as np
from scipy.io import loadmat, savemat
from spectral import *
import matplotlib.pyplot as plt
# input_data = loadmat('./data/Segundo/segundo.mat')['data']
input_data = loadmat('./data/Sandiego100/sandiego.mat')['data']
# input_data = loadmat('./data/Sandiego200/sandiego.mat')['data']
# input_data = loadmat('./data/Sandiego2/sandiego.mat')['data']
# input_data = loadmat('./data/Urban/urban.mat')['data']
# input_data = loadmat('./data/Beach/beach.mat')['data']
# input_data = loadmat('./data/HYDICE_urban/hydice_urban.mat')['data']
input_data = np.float32(input_data)
max1 = np.amax(input_data)
min1 = np.amin(input_data)
input_data = (input_data-min1)/(max1-min1)
pc = principal_components(input_data)
pc_099 = pc.reduce(fraction=0.99)
firstpca = pc_099.transform(input_data)
plt.imshow(firstpca[:, :, 1] , cmap='gray')     # Segundo=0, Sandiego100=1, Urban=0, Sandiego2=0, Beach=0, HYDICE=2
plt.axis('off')
plt.show()
# path = './guide_image_filter/fpcasegundo.mat'
# path = './guide_image_filter/fpcasandiego2.mat'
# path = './guide_image_filter/fpcasandiego100.mat'
# path = './guide_image_filter/fpcasandiego200.mat'
path = './guide_image_filter/fpcaurban.mat'
# path = './guide_image_filter/fpcabeach.mat'
# path = './guide_image_filter/fpcahydice.mat'
savemat(path, {'firstpca': firstpca})
# savemat(path, {'pca': firstpca[:,:,0]})

