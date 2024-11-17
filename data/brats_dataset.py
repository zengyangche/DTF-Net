from collections import defaultdict
from time import time
from data import BaseDataset
from data.nii_data_loader import nii_slides_loader, load_set, normalize_nii
import os
import os.path
import numpy as np
import cv2
import torch
import pickle


class BratsDataset(BaseDataset):
    def __init__(self, opt):
        '''
        Args:
        '''
        # 1. load form nii file
        if opt.isTrain:
            self.isTrain = True
            data_root = opt.dataroot
        else:
            self.isTrain = False
            data_root = opt.test_dataroot
        self.mode = opt.dataset_mode
        transform = normalize_nii
        loader = nii_slides_loader

        choose_slice_num = 78
        resize = 256

        flair_path = os.path.join(data_root, 'flair')
        t1_path = os.path.join(data_root, 't1')
        t1ce_path = os.path.join(data_root, 't1ce')
        t2_path = os.path.join(data_root, 't2')
        if self.isTrain:
            seg_path = os.path.join(data_root, 'seg')
            self.seg_set = load_set(seg_path)
        self.flair_set = load_set(flair_path)
        self.t1_set = load_set(t1_path)
        self.t1ce_set = load_set(t1ce_path)
        self.t2_set = load_set(t2_path)

        self.n_data = len(self.t1_set)
        
        # 2. create modal mask
        modal_names = self.get_modal_names()
        n_modal = len(modal_names)
        self.n_modal = n_modal

        # 3. load_all modal into memory
        print('Loading BraTS Dataset with "{}" mode...'.format(self.mode))
        start = time()
        cache_path = os.path.join(data_root, 'cache.pkl')
        if os.path.exists(cache_path):
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('load data from raw')
            self.data_dict = defaultdict(list)
            for index in range(self.n_data):
                print(index)
                if self.isTrain:
                    for modal in ['t1ce', 't1', 't2', 'flair', 'seg', 'texture']:
                        for i in range(choose_slice_num - 8 , choose_slice_num + 9, 2):  #         
                            if modal == 'texture':
                                modal_path, modal_target = getattr(self, 't1ce_set')[index]
                                modal_img = loader(modal_path, num=i, transform=transform)  # np.ndarray, shape=[224,224]
                                # Sobel
                                sobel_x = cv2.Sobel(modal_img, cv2.CV_64F, 1, 0, ksize=7)
                                sobel_y = cv2.Sobel(modal_img, cv2.CV_64F, 0, 1, ksize=7)
                                gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                                gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
                                normalized_img = gradient_magnitude_normalized / 255.0
                                modal_img = np.float32(normalized_img)
                                # modal_img = np.uint8(gradient_magnitude_normalized)
                                modal_img = cv2.resize(modal_img, (resize, resize))
                            else:
                                modal_path, modal_target = getattr(self, modal+'_set')[index]
                                if modal == 'seg':
                                    modal_img = loader(modal_path, num=i, transform=None)
                                else:
                                    modal_img = loader(modal_path, num=i, transform=transform) # np.ndarray, shape=[224,224]
                                modal_img = cv2.resize(modal_img, (resize, resize))
                            self.data_dict[modal].append(modal_img)
                else:
                    for modal in ['t1ce', 't1', 't2', 'flair', 'texture']:
                        for i in range(choose_slice_num, choose_slice_num+1, 1):
                            if modal == 'texture':
                                modal_path, modal_target = getattr(self, 't1ce_set')[index]
                                modal_img = loader(modal_path, num=i, transform=transform)  # np.ndarray, shape=[224,224]
                                # Sobel
                                sobel_x = cv2.Sobel(modal_img, cv2.CV_64F, 1, 0, ksize=7)
                                sobel_y = cv2.Sobel(modal_img, cv2.CV_64F, 0, 1, ksize=7)
                                gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                                gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
                                normalized_img = gradient_magnitude_normalized / 255.0
                                modal_img = np.float32(normalized_img)
                                # modal_img = np.uint8(gradient_magnitude_normalized)
                                modal_img = cv2.resize(modal_img, (resize, resize))
                            else:
                                modal_path, modal_target = getattr(self, modal+'_set')[index]
                                print(modal_path)
                                modal_img = loader(modal_path, num=i, transform=transform) # np.ndarray, shape=[224,224]
                                modal_img = cv2.resize(modal_img, (resize, resize))
                            self.data_dict[modal].append(modal_img)
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, segment_mask, texture) where target is class_index of the target class.
        """
        modal_order = ['flair', 't1', 't2', 'seg', 'texture', 't1ce']
        input_modal_names = modal_order[0:3]
        target_modal_name = modal_order[-1]
        seg_modal_name = modal_order[-3]
        texture_modal_name = modal_order[-2]

        A = []
        for modal_name in input_modal_names:
            modal_numpy = self.data_dict[modal_name][index]
            A.append(torch.tensor(modal_numpy[None], dtype=torch.float))
        # get ith target modal image array
        target_modal_numpy = self.data_dict[target_modal_name][index]
        texture_modal_numpy = self.data_dict[texture_modal_name][index]

        input = {
            'A': torch.cat(A),
            'B': torch.tensor(target_modal_numpy[None], dtype=torch.float),
            'T': torch.tensor(texture_modal_numpy[None], dtype=torch.float),
            'modal_names': modal_order
        }
        if self.isTrain:
            seg_modal_numpy = self.data_dict[seg_modal_name][index]
            input['S'] = torch.tensor(seg_modal_numpy[None].astype(np.uint8), dtype=torch.uint8)

        return input

    def __len__(self):
        if self.isTrain:
            return self.n_data * 9
        else:
            return self.n_data
    def get_modal_names(self):
        return ['flair', 't1', 't2', 't1ce', 'seg', 'texture']
