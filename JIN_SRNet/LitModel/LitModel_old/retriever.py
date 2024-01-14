import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import jpegio as jio
import pandas as pd
import numpy as np
import pickle
# import cv2
import albumentations as A
import os
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.insert(1,'./')
# from tools.jpeg_utils import *
# import h5py


def get_train_transforms(size=512):
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def get_valid_transforms(size=512):
    return A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class TrainRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None, return_name=False):
        super().__init__()
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.mean = np.array([0.3914976, 0.44266784, 0.46043398])
        self.std = np.array([0.17819773, 0.17319807, 0.18128773])

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp)
            if tmp.image_components==1:
                image = np.repeat(image, 3, -1).astype(np.float32)
            else:
                image = ycbcr2rgb(image).astype(np.float32)
            image /= 255.0
        elif self.decoder == 'RJCA_color':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp).astype(np.float32)
            if tmp.image_components==1:
                image = np.repeat(image, 3, -1).astype(np.float32)
            else:
                image = ycbcr2rgb(image).astype(np.float32)
            image = image - np.round(image)
        elif self.decoder == 'RJCA_Y':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp)[:,:,0].astype(np.float32)
#             if tmp.image_components==1:
#                 image = np.repeat(image, 3, -1).astype(np.float32)
#             else:
#                 image = ycbcr2rgb(image).astype(np.float32)
            image = image - np.round(image)
        elif self.decoder == 'eY':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp)[:,:,0].astype(np.float32)
#             if tmp.image_components==1:
#                 image = np.repeat(image, 3, -1).astype(np.float32)
#             else:
#                 image = ycbcr2rgb(image).astype(np.float32)
            e = image - np.round(image)
            image = np.stack([image/255, e], axis=-1)
        else:
            image = cv2.imread(f'{self.data_path}/{kind}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(2, label)
        
        if self.return_name:
            return image, target, image_name
        else:
            return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)

class TrainRetriever_pt(Dataset):
    """
    Represents a dataset which contains information about images saved as tensors of shape (1, 3, height, width).
    """

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None, return_name=False):
        super().__init__()
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.classes = len(np.unique(self.kinds))

        if not self.data_path.endswith("/"):
            self.data_path += "/"

    def __getitem__(self, index: int):
        kind, image_n, label = self.kinds[index], self.image_names[index], self.labels[index]

        if self.decoder == "NR":
            img_t = torch.load(self.data_path + image_n,map_location="cpu")
            image = img_t[0].permute(1,2,0).cpu().numpy()
        else:
            raise ValueError(f"The decoder {self.decoder} is not supported")
        
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]
        
        if self.return_name: 
            return image, label, image_n
        else: 
            return image, label
        
    def __len__(self) -> int:
        return self.image_names.shape[0]
    
    def get_labels(self):
        return list(self.labels)


class TrainRetriever_hdf5(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None, return_name=False):
        super().__init__()
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.mean = np.array([0.3914976, 0.44266784, 0.46043398])
        self.std = np.array([0.17819773, 0.17319807, 0.18128773])
        self.hdf5 = {}
        self.classes = len(np.unique(self.kinds))
        for kind in np.unique(self.kinds):
            if 'Cover' in kind:
                self.hdf5[kind] = h5py.File(data_path + kind + '.h5py', 'r')['cover']
                self.Q = h5py.File(data_path + kind + '.h5py', 'r')['Q'][:]
            else:
                self.hdf5[kind] = h5py.File(data_path + kind + '.h5py', 'r')['stego']
            
        #if 'Q' in self.hdf5['Cover'].keys():
        #    self.Q = self.hdf5['Cover']['Q'][:]
        #else:
        #    self.Q_Y = self.hdf5[kind]['Q_0'][:]
        #    self.Q_CbCr = self.hdf5[kind]['Q_1'][:]

    def __getitem__(self, index: int):
        
        kind, image_n, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        if  self.decoder == 'NR':
            C = self.hdf5[kind][image_n]
            image = decompress_matrix(C,self.Q)
            image = np.repeat(image, 3, -1).astype(np.float32)
            # if tmp.image_components==1:
            #     image = np.repeat(image, 3, -1).astype(np.float32)
            # else:
            #     image = ycbcr2rgb(image).astype(np.float32)
            image /= 255.0
        elif self.decoder == 'RJCA_color':
            C = self.hdf5[kind][image_n]
            image = decompress_matrix(C,self.Q).astype(np.float32)
            image = np.repeat(image, 3, -1)
            image = image - np.round(image)
        elif self.decoder == 'RJCA_color2':
            Y = self.hdf5[kind][image_name + '._0'][:]
            image_Y = decompress_matrix(Y,self.Q_Y).astype(np.float32)
            Cb = self.hdf5[kind][image_name + '._1'][:]
            image_Cb = decompress_matrix(Cb,self.Q_CbCr).astype(np.float32)
            Cr = self.hdf5[kind][image_name + '._2'][:]
            image_Cr = decompress_matrix(Cr,self.Q_CbCr).astype(np.float32)
            image = np.concatenate([image_Y, image_Cb, image_Cr], axis=-1)
            image = image - np.round(image)
        elif self.decoder == 'RJCA_chroma':
            Cb = self.hdf5[kind][image_name + '._1'][:]
            image_Cb = decompress_matrix(Cb,self.Q_CbCr).astype(np.float32)
            Cr = self.hdf5[kind][image_name + '._2'][:]
            image_Cr = decompress_matrix(Cr,self.Q_CbCr).astype(np.float32)
            image = np.concatenate([image_Cb, image_Cr], axis=-1)
            image = image - np.round(image)
        elif self.decoder == 'eCbCr':
            Cb = self.hdf5[kind][image_name + '._1'][:]
            image_Cb = decompress_matrix(Cb,self.Q_CbCr).astype(np.float32)
            Cr = self.hdf5[kind][image_name + '._2'][:]
            image_Cr = decompress_matrix(Cr,self.Q_CbCr).astype(np.float32)
            image = np.concatenate([image_Cb, image_Cr], axis=-1)
            error = image - np.round(image)
            image = np.stack([image/255, error], axis=-1)
        elif self.decoder == 'RJCA_Y':
            C = self.hdf5[kind][image_n]
            image = decompress_matrix(C,self.Q).astype(np.float32)
            image = image - np.round(image)
        elif self.decoder == 'eY':
            C = self.hdf5[kind][image_n]
            image = decompress_matrix(C,self.Q).astype(np.float32)
            e = image - np.round(image)
            image = np.concatenate([image/255, e], axis=-1)
        else:
            image = cv2.imread(f'{self.data_path}/{kind}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            

        
        if self.return_name:
            return image, label, image_n
        else:
            return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
class TrainRetrieverPaired(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        i = np.random.choice([1,2,3])
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.data_path}/{kind[0]}/{image_name[0]}')
            cover = decompress_structure(tmp)
            cover = ycbcr2rgb(cover).astype(np.float32)
            cover /= 255.0
            tmp = jio.read(f'{self.data_path}/{kind[i]}/{image_name[i]}')
            stego = decompress_structure(tmp)
            stego = ycbcr2rgb(stego).astype(np.float32)
            stego /= 255.0
        else:
            cover = cv2.imread(f'{self.data_path}/{kind[0]}/{image_name[0]}', cv2.IMREAD_COLOR)
            cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB).astype(np.float32)
            cover /= 255.0
            stego = cv2.imread(f'{self.data_path}/{kind[i]}/{image_name[i]}', cv2.IMREAD_COLOR)
            stego = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB).astype(np.float32)
            stego /= 255.0
            
        target_cover = onehot(4, label[0])
        target_stego = onehot(4, label[i])
            
        if self.transforms:
            sample = {'image': cover, 'image2': stego}
            sample = self.transforms(**sample)
            cover = sample['image']
            stego = sample['image2']
            
        return torch.stack([cover,stego]), torch.stack([target_cover, target_stego])

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class TestRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None):
        super().__init__()
        self.data_path = data_path
        self.test_data_path = self.data_path+'Test/'
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.test_data_path}/{image_name}')
            image = decompress_structure(tmp).astype(np.float32)
            image = ycbcr2rgb(image)
            image /= 255.0
        else:
            image = cv2.imread(f'{self.folder}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            
        image = self.func_transforms(image)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]
    
    
class TrainRetrieverOHPaired(Dataset):
    def __init__(self, data_path, kinds, image_names, labels, num_classes=2, decoder='NR', transforms=True, T=5):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.num_classes = num_classes
        self.transforms = transforms
        self.T = T
        
    def __preprocess_strcture(self, struct, label, rot, flip):
        struct = rot_and_flip_jpeg(struct, rot, flip)
        image_dct = np.dstack(struct.coef_arrays).astype(np.float32)
        image = decompress_structure(struct).astype(np.float32)
        image = torch.from_numpy(image.transpose(2,0,1))
        
        image_dct = abs_bounded_onehot(image_dct, T=self.T).astype(np.float32)
        image_dct = torch.from_numpy(image_dct.transpose(2,0,1))
        
        label = onehot(2, label)
        return image, image_dct, label
    
    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        if self.transforms:
            rot = np.random.randint(0,4)
            flip = np.random.rand() < 0.5
        else:
            rot = 0
            flip = False
        path = f'{self.data_path}/{kind[0]}/{image_name}'
        tmp = jio.read(str(path))
        cover, cover_dct, target_cover = self.__preprocess_strcture(tmp, label[0], rot, flip)
        
        path = f'{self.data_path}/{kind[1]}/{image_name}'
        tmp = jio.read(str(path))
        stego, stego_dct, target_stego = self.__preprocess_strcture(tmp, label[1], rot, flip)

        return torch.stack([cover,stego]), torch.stack([cover_dct,stego_dct]), torch.stack([target_cover, target_stego])

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
