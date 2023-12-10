# -*- coding: utf-8 -*-
# Author: Xi Chen
# Mailbox: 18742530685@163.com
# Warning: If you use this code, please refer to this paper  <<Meta Learning for Hyperspectral Target Detection using Siamese Network>>


from typing import List

import numpy as np 
import torch
import os
import torch.utils.data as data
from scipy import io
from networks import TripletNet, FECNN
from utils import cos_sim, HyperX
from utils import plot_roc_curve
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






# Sandiego2  189bands   AVIRIS
data_name = 'Sandiego2'
hyperdata_path = './data/Sandiego2/sandiego.mat'
dgt_path = './data/Sandiego2/groundtruth.mat'
prior_path = './data/Sandiego2/tgt_sandiego_2.mat'

hyperdata = io.loadmat(hyperdata_path)['data']
dgt = io.loadmat(dgt_path)['gt']
prior = io.loadmat(prior_path)['tgt']




# ...............................................................................................
hyperdata = np.float32(hyperdata)
max1 = np.amax(hyperdata)
min1 = np.amin(hyperdata)
hyperdata = (hyperdata-min1)/max1

detect_dataset = HyperX(hyperdata, dgt)
detect_loader = data.DataLoader(detect_dataset, batch_size=1)


'''
channe = hyperdata.shape[2]
hyperdata = hyperdata.reshape(-1, channe)
'''
prior = np.float32(prior)
max2 = np.amax(prior)
min2 = np.amin(prior)
prior = (prior-min2)/max2
'''
dgt = dgt.astype(np.float32)
dgt = dgt.reshape(-1)
'''

model = TripletNet(FECNN()).to(device)
#model.load_state_dict(torch.load('decomposed_model.ckpt').module.state_dict())
model.load_state_dict(torch.load('model.ckpt'))


#model = torch.load("model").cuda()
#model = torch.load("decomposed_model").cuda()


prior = np.expand_dims(prior, axis=0)

prior = torch.from_numpy(prior)
# print(prior.shape)


model.eval()
target_detector: List[None] = []
sandiego_feature: List[None] = []
with torch.no_grad():
    prior = prior.to(device)
    prior_output = model.get_embedding(prior)
    prior_output = prior_output.cuda().data.cpu().numpy()
    for images, _ in detect_loader:
        images = torch.squeeze(images)
        images = torch.unsqueeze(images, 0)
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        outputs = model.get_embedding(images)
        outputs = outputs.cuda().data.cpu().numpy()
        detection = cos_sim(prior_output, outputs)
        target_detector.append(detection)
        sandiego_feature.append(outputs)

target_detector = np.array(target_detector)
target_detector = target_detector.squeeze()

sandiego_feature = np.array(sandiego_feature)
sandiego_feature = sandiego_feature.squeeze()

tdgt = dgt.astype(np.float32)
tdgt = tdgt.reshape(-1)

max3 = np.amax(target_detector)
min3 = np.amin(target_detector)
target_detector = (target_detector - min3)/(max3 - min3)

plot_roc_curve(tdgt, target_detector, data_name)

H, W = dgt.shape
target_detector = np.reshape(target_detector, (H, W))

plt.figure(2)
# plt.imshow(target_detector)
plt.imshow(target_detector, cmap='gray')
plt.axis('off')
pathfigure = './result/' + data_name + '.png'
plt.savefig(pathfigure, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
plt.show()
path = './result/' + data_name + '.mat'
pathh = './guide_image_filter/' + data_name + '.mat'
io.savemat(path, {'detect': target_detector})
io.savemat(pathh, {'detect': target_detector})

sandiego_feature_path = './result/' + 'sandiego_feature' + '.mat'
io.savemat(sandiego_feature_path,{'sandiego_feature': sandiego_feature})





'''
model.eval()
target_detector: List[None] = []
with torch.no_grad():
    prior = prior.to(device)
    prior_output = model(prior)
    prior_output = prior_output.cuda().data.cpu().numpy()
    for images, _ in detect_loader:
        images = torch.squeeze(images)
        images = torch.unsqueeze(images, 0)
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.cuda().data.cpu().numpy()
        detection = cos_sim(prior_output, outputs)
        target_detector.append(detection)

target_detector = np.array(target_detector)
target_detector = target_detector.squeeze()
data_name = 'Sandiego100'
tdgt = dgt.astype(np.float32)
tdgt = tdgt.reshape(-1)
plot_roc_curve(tdgt, target_detector, data_name)
H, W = dgt.shape
target_detector = np.reshape(target_detector, (H, W))
plt.figure(2)
plt.imshow(target_detector)
plt.show()
path = './result/' + data_name + '.mat'
io.savemat(path, {'detect': target_detector})
'''


'''
from fine_tuning import FTmodel
Model = FTmodel(model).to(device)
#model.load_state_dict(torch.load('decomposed_model.ckpt').module.state_dict())
Model.load_state_dict(torch.load('FTmodel.ckpt'))


#model = torch.load("model").cuda()
#model = torch.load("decomposed_model").cuda()


prior = np.expand_dims(prior, axis=0)

prior = torch.from_numpy(prior)
# print(prior.shape)


Model.eval()
target_detector: List[None] = []
with torch.no_grad():
    prior = prior.to(device)
    prior_output = Model(prior)
    prior_output = prior_output.cuda().data.cpu().numpy()
    for images, _ in detect_loader:
        images = torch.squeeze(images)
        images = torch.unsqueeze(images, 0)
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        outputs = Model(images)
        outputs = outputs.cuda().data.cpu().numpy()
        detection = cos_sim(prior_output, outputs)
        target_detector.append(detection)

target_detector = np.array(target_detector)
target_detector = target_detector.squeeze()

tdgt = dgt.astype(np.float32)
tdgt = tdgt.reshape(-1)
plot_roc_curve(tdgt, target_detector, data_name)
H, W = dgt.shape
target_detector = np.reshape(target_detector, (H, W))
plt.figure(2)
plt.imshow(target_detector)
plt.axis('off')
pathfigure = './result/' + data_name + '.png'
plt.savefig(pathfigure, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
plt.show()
path = './result/' + data_name + '.mat'
io.savemat(path, {'detect': target_detector})
'''



