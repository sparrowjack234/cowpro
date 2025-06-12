"""
Dataset for training with pseudolabels
TODO:
1. Merge with manual annotated dataset
2. superpixel_scale -> superpix_config, feed like a dict
"""
import glob
import numpy as np
import dataloaders.augutils as myaug
import torch
import random
import os
import copy
import platform
import json
import re
from dataloaders.common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList
import dataloaders.augutils as myaug
import torchvision.transforms as deftfx
import dataloaders.image_transforms as myit

from skimage import segmentation, color
from skimage.filters import sobel
from skimage import graph
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import ast

# from graph_merge import *

class edge:
    def __init__(self, x,y):
        self.x = x
        self.y = y

class SuperpixelDataset(BaseDataset):
    def __init__(self, 
                which_dataset, 
                base_dir, 
                idx_split, 
                mode, scan_per_load, transforms = None, transform_param_limits = None, 
                num_rep = 2, 
                min_fg = '', nsup = 1, 
                fix_length = None, tile_z_dim = 3, 
                exclude_list = [], seg_method = 'ncuts', 
                superpix_scale = 'MIDDLE',
                dataset_config = None, **kwargs):
        """
        Pseudolabel dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            nsup:               number of scans used as support. currently idle for superpixel dataset
            transforms:         data transform (augmentation) function
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            num_rep:            Number of augmentation applied for a same pseudolabel
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            superpix_scale:     config of superpixels
        """
        super(SuperpixelDataset, self).__init__(base_dir)

        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.pseu_label_name = DATASET_INFO[which_dataset]['PSEU_LABEL_NAME']
        self.real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

        self.dataset_config = dataset_config[which_dataset]
       

        self.transform_param_limits = transform_param_limits
        


        self.is_train = True if mode == 'train' else False
        assert mode == 'train'
        self.fix_length = fix_length
        self.nclass = len(self.pseu_label_name)
        self.num_rep = num_rep
        self.tile_z_dim = tile_z_dim

        # find scans in the data folder
        self.nsup = nsup
        self.base_dir = base_dir
        # print(self.base_dir)
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        # print(self.img_pids)
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x)))

        # experiment configs
        self.exclude_lbs = ast.literal_eval(exclude_list) if isinstance(exclude_list, str) else exclude_list
        # print(self.exclude_lbs)
        self.superpix_scale = superpix_scale
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')
        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)
        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold
        self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])

        if self.is_train:
            if scan_per_load > 0: # if the dataset is too large, only reload a subset in each sub-epoch
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        else:
            raise Exception
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()

        print("###### Initial scans loaded: ######")
        print(self.pid_curr_load)

        print("#### Setting up augmentation population ####")
        self.affine     = self.transform_param_limits.get('affine', 0)
        self.alpha      = self.transform_param_limits.get('elastic',{'alpha': 0})['alpha']
        self.sigma      = self.transform_param_limits.get('elastic',{'sigma': 0})['sigma']
        self.gamma_range = self.transform_param_limits['gamma_range']

        self.randomaffine = myit.RandomAffine(self.affine.get('rotate'),
                                             self.affine.get('shift'),
                                             self.affine.get('shear'),
                                             self.affine.get('scale'),
                                             self.affine.get('scale_iso',True),
                                             order=3)

        self.elastic = myit.ElasticTransform(self.alpha, self.sigma)

        self.sup_max_cls = 0


    def get_scanids(self, mode, idx_split):
        """
        Load scans by train-test split
        leaving one additional scan as the support scan. if the last fold, taking scan 0 as the additional one
        Args:
            idx_split: index for spliting cross-validation folds
        """
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a only portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing notiong")
            return -1

        del self.actual_dataset
        del self.info_by_scan

        self.pid_curr_load = np.random.choice( self.scan_ids, size = self.scan_per_load, replace = False )
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            ####### comment out for agun model ########
            _lb_fid  = os.path.join(self.base_dir, f'superpix-{self.superpix_scale}_{curr_id}.nii.gz')
            ##########################################

            curr_dict["img_fid"] = _img_fid
            ######## comment out for agun model #######
            curr_dict["lbs_fid"] = _lb_fid
            ###########################################
            out_list[str(curr_id)] = curr_dict
        return out_list

    def read_dataset(self):
        """
        Read images into memory and store them in 2D
        Build tables for the position of an individual 2D slice in the entire dataset
        """
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for scan_id, itm in self.img_lb_fids.items():
            # print(scan_id, itm)
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
            # print(img.shape)
            img = img.transpose(1,2,0)
            # print(img.shape)
            self.info_by_scan[scan_id] = _info

            img = np.float32(img)
            img, mean, std  = self.norm_func(img)

            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]
            ######## comment out for agun model #########
            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1,2,0)
            lb = np.int32(lb)
            #############################################

            img = img[:256, :256, :]
            lb = lb[:256, :256, :]

            # format of slices: [axial_H x axial_W x Z]

            # assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            # re-organize 3D images into 2D slices and record essential information for each slice
            out_list.append( {"img": img[..., 0: 1],
                           "lb":lb[..., 0: 0 + 1],
                           "mean":mean,
                           "std":std,
                           "sup_max_cls": lb[..., 0: 0 + 1].max(),
                           "is_start": True,
                           "is_end": False,
                           "nframe": img.shape[-1],
                           "scan_id": scan_id,
                           "z_id":0})

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii + 1],
                           "mean":mean,
                           "std":std,
                           "is_start": False,
                           "is_end": False,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last slice of a 3D volume
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii+ 1],
                           "mean":mean,
                           "std":std,
                           "is_start": False,
                           "is_end": True,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

            # print(len(out_list))

        return out_list

    def read_classfiles(self):
        """
        Load the scan-slice-class indexing file
        """
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_100.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def supcls_pick_binarize(self, super_map, sup_max_cls, bi_val = None): # 
        """
        pick up a certain super-pixel class or multiple classes, and binarize it into segmentation target
        Args:
            super_map:      super-pixel map
            bi_val:         if given, pick up a certain superpixel. Otherwise, draw a random one
            sup_max_cls:    max index of superpixel for avoiding overshooting when selecting superpixel

        """
        possible_values = np.sort(np.unique(super_map))[1:]
        if bi_val == None:
            bi_val = random.choice(possible_values) #int(torch.randint(low = 1, high = int(sup_max_cls), size = (1,)))
        return np.float32(super_map == bi_val)

        ################################################


    def gamma_transform(self, img):
        # gamma_range = aug['aug']['gamma_range']
        if isinstance(self.gamma_range, tuple):
            gamma = np.random.rand() * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif self.gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(self.gamma_range))
        return img, gamma


    def random_cutmix(self, img, lb):
        h = w = 0
        cx = cy = None
        for i in range(lb.shape[0]):
            if lb[i,:,:].sum() > 0:
                if cx is None:
                    cx = i
                h+=1
            elif cx is not None:
                break

        for j in range(lb.shape[1]):
            if lb[:,j,:].sum() > 0:
                if cy is None:
                    cy = j
                w+=1
            elif cy is not None:
                break

        # print(cx,cy,h,w)

        if cx is not None and cy is not None and h!=0 and w!=0:
            dx = int(random.random()*(img.shape[0]-h))
            dy = int(random.random()*(img.shape[1]-w))
            tmp = img[dx:dx+h,dy:dy+w]
            img[cx:cx+h,cy:cy+w] = img[dx:dx+h,dy:dy+w]
            img[dx:dx+h,dy:dy+w] = tmp

            tmp = lb[dx:dx+h,dy:dy+w]
            lb[cx:cx+h,cy:cy+w] = lb[dx:dx+h,dy:dy+w]
            lb[dx:dx+h,dy:dy+w] = tmp

        return img, lb

    def random_erasing(self, img, lb):
        h = w = 0
        cx = cy = None
        for i in range(lb.shape[0]):
            if lb[i,:,:].sum() > 0:
                if cx is None:
                    cx = i
                h+=1
            elif cx is not None:
                break

        for j in range(lb.shape[1]):
            if lb[:,j,:].sum() > 0:
                if cy is None:
                    cy = j
                w+=1
            elif cy is not None:
                break

        # print(cx,cy,h,w)

        if cx is not None and cy is not None and h!=0 and w!=0:
            img[cx:cx+h,cy:cy+w] = np.zeros_like(img[cx:cx+h,cy:cy+w])
            lb[cx:cx+h,cy:cy+w] = np.zeros_like(lb[cx:cx+h,cy:cy+w])

        return img, lb

    def transform_img_lb(self, comp, c_label, c_img, use_onehot, nclass, num_rep, **kwargs):
        """
        Args
        comp:               a numpy array with shape [H x W x C + c_label]
        c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
        nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label

        """
        params = []

        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        # geometric transform
        _label = comp[..., c_img ]
        _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
        comp = np.concatenate( [comp[...,  :c_img ], _h_label], -1 )
        ########### AFFINE TRANSFOMRATIONS ################
        affine_params = self.randomaffine.build_M(comp.shape[:2])
        comp = self.randomaffine(comp, affine_params)
        affine_params = torch.from_numpy(affine_params.flatten())
        ##################################################
        ########### ELASTIC TRANSFORMATION ###############
        comp, dx_params, dy_params = self.elastic(comp)
        dx_params = torch.from_numpy(dx_params.flatten())
        dy_params = torch.from_numpy(dy_params.flatten())
        ##################################################
        # comp = geometric_tfx(comp)
        ##################################################
        # round one_hot labels to 0 or 1
        t_label_h = comp[..., c_img : ]
        t_label_h = np.rint(t_label_h)
        assert t_label_h.max() <= 1
        t_img = comp[..., 0 : c_img ]
        ############## intensity transform ################
        t_img, gamma = self.gamma_transform(t_img)
        gamma = torch.Tensor([gamma])

        params = torch.cat([affine_params, dx_params, dy_params, gamma])
        ##################################################
        # if use_onehot is True:
        #     t_label = t_label_h
        # else:
        t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)

        label_remove = False

        return t_img, t_label, params, label_remove


    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        # ============ GETS THE IMAGES AND LABELS AND OTHER INFOS ========== #
        curr_dict = self.actual_dataset[index]
        sup_max_cls = curr_dict['sup_max_cls']
        if sup_max_cls < 1:
            return self.__getitem__(index + 1)

        image_t = curr_dict["img"]
        label_raw = curr_dict["lb"]
        # print(image_t.shape)
        # ================================================================= #
        # if using setting 2, this slice need to be excluded since it contains label which is supposed to be unseen
        # if using setting 1, this slice need not be excluded since it contains label which is not suppossed to be unseen
        for _ex_cls in self.exclude_lbs:
            if curr_dict["z_id"] in self.tp1_cls_map[self.real_label_name[_ex_cls]][curr_dict["scan_id"]]:
                return self.__getitem__(torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))

        # =================== KMEANS SUPERPIXEL METHOD ============#
        # try:
        label_t = self.supcls_pick_binarize(label_raw, sup_max_cls, bi_val = None) #image_t)  #comment out for agun model
        # except:
            # return self.__getitem__(torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))
        # plt.imshow(label_t, vmin = 0, vmax = 1.0, cmap = 'gray')
        # plt.title('Label')
        # plt.show()
        ############ comment out for agun model ######
        # label_t = label_t[:,:,None]
        ###############################################
        #==========================================================#

        pair_buffer = []

        comp = np.concatenate( [curr_dict["img"], label_t], axis = -1 )

        support_flag = True

        for ii in range(self.num_rep):
            # ============= TRANSFORMATONS ================================ #
            # img, lb = self.transforms(comp, c_img = 1, c_label = 1, nclass = self.nclass,  is_train = True, use_onehot = False)
            ################### comment off for agun model ##########
            img, lb, tr_params, label_remove = self.transform_img_lb(comp, c_img = 1, c_label = 1, nclass = self.nclass,  is_train = True, use_onehot = False, num_rep = ii)
            # tr_params = [] #None
            # -----------------------------------------
            #########################################################
            if img.ndim == 2:
                img = img[:,:,None]
            img = torch.from_numpy( np.transpose( img, (2, 0, 1)) )
            # print(img.shape)
            # lb  = torch.from_numpy( lb.transpose((2,0,1)))
            lb  = torch.from_numpy( lb.squeeze(-1))
            # print(lb.shape)

            if self.tile_z_dim:
                img = img.repeat( [ self.tile_z_dim, 1, 1] )
                # lb = lb.repeat([self.tile_z_dim, 1, 1])
                assert img.ndimension() == 3, f'actual dim {img.ndimension()}'
            # print(img.shape)

            is_start = curr_dict["is_start"]
            is_end = curr_dict["is_end"]
            nframe = np.int32(curr_dict["nframe"])
            scan_id = curr_dict["scan_id"]
            z_id    = curr_dict["z_id"]

            sample = {"image": img,
                    "label":lb,
                    "mean":curr_dict["mean"],
                    "std":curr_dict["std"],
                    "is_start": is_start,
                    "is_end": is_end,
                    "nframe": nframe,
                    "scan_id": scan_id,
                    "z_id": z_id,
                    'params':tr_params
                    }

            # Add auxiliary attributes
            if self.aux_attrib is not None:
                for key_prefix in self.aux_attrib:
                    # Process the data sample, create new attributes and save them in a dictionary
                    aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                    for key_suffix in aux_attrib_val:
                        # one function may create multiple attributes, so we need suffix to distinguish them
                        sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]
            pair_buffer.append(sample)



        support_images = []
        support_mask = []
        support_class = []

        query_images = []
        query_labels = []
        query_class = []

        support_params = []
        query_params = []

        for idx, itm in enumerate(pair_buffer):
            if idx % 2 == 0:
                support_images.append(itm["image"])
                support_class.append(1) # pseudolabel class
                support_mask.append(  self.getMaskMedImg( itm["label"], 1, [1]  ))
                support_params.append(itm['params'])

            else:
                query_images.append(itm["image"])
                query_class.append(1)
                query_labels.append(  itm["label"])
                query_params.append(itm['params'])

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
            'query_images': query_images, #
            'query_labels': query_labels,
            'support_params' : support_params,
            'query_params' : query_params,
            "mean":curr_dict["mean"],
            "std":curr_dict["std"]
        }


    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length != None:
            
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
            class_ids:      all class id in this episode
        """
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}
