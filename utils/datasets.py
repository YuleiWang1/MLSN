import numpy as np
from PIL import Image
import torch
from scipy import io 
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler



class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, train_data, train_label, train_zero, train=True):
        self.train = train


        
        if self.train:
            self.train_labels = train_label
            #self.train_labels = self.train_labels[self.train_labels != 0]
            #self.train_labels = filter(lambda x: x != 0, self.train_labels)
            #self.train_labels = self.train_labels.astype(np.float32)
            #self.train_labels = list(self.train_labels)
            self.train_data = train_data
            self.train_label0 = train_zero

            self.labels_set = set(self.train_label0)
            '''
            self.labels_set =set()
            for item in self.train_labels:
                for i in item:
                    self.labels_set.add(i)
            '''
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
            #self.label_index = np.array([np.where(self.train_labels == label)[0] for label in self.labels_set])
            self.label_index = np.nonzero(self.train_labels)
            self.label_index = np.array([x for x in zip(self.label_index)])
            self.label_index = np.squeeze(self.label_index)
            
            


        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            index = self.label_index[index]
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
        '''
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        '''
        '''
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        '''
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img3 = np.expand_dims(img3, axis=0)
        return (img1, img2, img3), []
    
    def __len__(self):
        return len(self.train_label0)#len(self.mnist_dataset)
    

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def load_salinas_mat(data_path, gt_path, remove_bands=True):
    data = io.loadmat(data_path)['salinas'].astype(np.float32)
    labels = io.loadmat(gt_path)['salinas_gt']

    # Delete several bands for keeping the the same bands with testing dataset (from 224 decrease to 189)
    if remove_bands:


        # 189bands
        bands = np.concatenate((np.arange(6, 32),
                                   np.arange(35, 96),
                                   np.arange(97, 106),
                                   np.arange(113, 152),
                                   np.arange(166, 220)))
        data = data[:, :, bands]

        # 175bands
        # bands = np.concatenate((np.arange(6, 30),
        #                         np.arange(35, 93),
        #                         np.arange(97, 103),
        #                         np.arange(113, 149),
        #                         np.arange(166, 217)))
        # data = data[:, :, bands]

        # 204bands
        # bands = np.concatenate((np.arange(3, 32),
        #                         np.arange(33, 96),
        #                         np.arange(97, 106),
        #                         np.arange(109, 152),
        #                         np.arange(160, 220)))
        # data = data[:, :, bands]

        # 126bands
        # bands = np.concatenate((np.arange(16, 32),
        #                         np.arange(45, 96),
        #                         np.arange(97, 106),
        #                         np.arange(113, 142),
        #                         np.arange(169, 190)))
        # data = data[:, :, bands]






    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels 

