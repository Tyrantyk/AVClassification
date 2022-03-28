
import numpy as np
import os
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AVDRIVEloader(torch.utils.data.Dataset):
    def __init__(self, dir_img=None, dir_gt=None, dir_ves=None, dir_gabor=None, dir_linear=None):
        super(AVDRIVEloader, self).__init__()

        self.dir_img = dir_img
        self.dir_gt = dir_gt
        #self.dir_gabor = dir_gabor
        self.dir_ves = dir_ves
        #self.dir_linear = dir_linear

        
        fn_img = os.listdir(dir_img)
        fn_gt = os.listdir(dir_gt)
        #fn_gabor = os.listdir(dir_gabor)
        #fn_linear = os.listdir(dir_linear)
        fn_ves = os.listdir(dir_ves)
 
        
        fn_img.sort()
        fn_gt.sort()
        #fn_linear.sort()
        fn_ves.sort()
        #fn_gabor.sort()
 
        assert len(fn_img) == len(fn_gt)
        #assert len(fn_img) == len(fn_gabor)
        assert len(fn_img) == len(fn_ves)
        #assert len(fn_img) == len(fn_linear)
 

        idx = [*range(0, len(fn_img))]
        self.fn_img = [fn_img[i] for i in idx]
        self.fn_gt = [fn_gt[i] for i in idx]
        #self.fn_gabor = [fn_gabor[i] for i in idx]
        self.fn_ves = [fn_ves[i] for i in idx]
        #self.fn_linear = [fn_linear[i] for i in idx]

        self.nums_img = len(self.fn_img)

    def __len__(self):
        return len(self.fn_img) 

    def __getitem__(self, idx):
        img_name = self.fn_img[idx]
        gt_name = self.fn_gt[idx]
        #gabor_name = self.fn_gabor[idx]
        ves_name = self.fn_ves[idx]
        #linear_name = self.fn_linear[idx] 
        img = self.img_process(self.dir_img, img_name, type='img')
        gt = self.img_process(self.dir_gt, gt_name, type='gt')
        #gabor = self.img_process(self.dir_gabor, gabor_name, type='gabor')
        #linear = self.img_process(self.dir_linear, linear_name, type='linear')
        ves = self.img_process(self.dir_ves, ves_name, type='vessel')
        #img = torch.cat((img,gabor),0)
        #img = torch.cat((img,linear),0)
        #img_pad = torch.zeros((3,640,640))
        #img_pad[:,28:(28+584),37:(37+565)] = img
        #gt_pad = torch.zeros((640,640))
        #gt_pad[28:(28+584),37:(37+565)] = gt
        #ves_pad = torch.zeros((1,640,640))
        #ves_pad[:,28:(28+584),37:(37+565)] = ves
        return img, gt, ves

    def img_process(self, dir_, img_name, type):
        img = Image.open(os.path.join(dir_, img_name))
        img_np = np.array(img)
        
        if type=='img':
            img_np = np.transpose(img_np, (2,0,1))
            img_np = np.array(img_np / 255., dtype=np.float32)
            if '64' in str(img_np.dtype):
                print(img_np.dtype)
        elif type=='gt':
            assert len(img_np.shape)==2
        elif type=='gabor' or type=='vessel' or type=='linear':
            img_np = np.expand_dims(img_np, axis=0)
            if np.max(img_np)>1:
                img_np = img_np / 255. 

        return torch.tensor(img_np)



