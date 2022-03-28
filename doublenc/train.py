import numpy as np
import os
import torchvision.transforms
from model import *
from DataLoader import *
import argparse
import time
from new_test import *
from ema import *
from layer_name import *
import random
from PIL import Image
from models.resnet_simclr import ResNetSimCLR
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dir_avdrive_train_img = "../data/training/images_aug_patch"
dir_avdrive_train_gt = "../data/training/av_label_aug_patch"
dir_avdrive_train_skeleton = "../data/training/skeleton_aug_patch"
dir_avdrive_train_vessel = "../data/training/vessel_aug_patch"
dir_avdrive_test_img = "../data/test/images"
dir_avdrive_test_gt = "../data/test/av_all"
dir_avdrive_test_skeleton = "../data/test/skeleton"
dir_avdrive_test_vessel = "../data/test/vessel"

parser = argparse.ArgumentParser(description='A/V classification')
parser.add_argument('-l', default=0.5, type=float,
                    help='ratio of ce and bce loss')
parser.add_argument('-epoch', default=1000, type=int)
parser.add_argument('-batch_size', default=16, type=float)
parser.add_argument('-seed', default=20, type=int,
                    help='random seed')
args = parser.parse_args()
lossf_ce = torch.nn.CrossEntropyLoss(ignore_index=3)
lossf_bce = torch.nn.BCELoss()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

max_epoch=args.epoch
batch_size=args.batch_size
l = args.l

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def train_2task_AVDRIVE(dataset='AVDRIVE'):

    # inspire_dataset = INSPIREloader.INSPIREloader_ALLTEST(dir_img, dir_gt, dir_ske)
    # inspire_testloader = torch.utils.data.DataLoader(inspire_dataset, batch_size=1, shuffle=False)
    
    if dataset == 'AVDRIVE':
        trainloader = avdrive_trainloader
        testloader = avdrive_testloader

    simclr = ResNetSimCLR('resnet34', 128).cuda()
    dic = torch.load('checkpoint_0200.pth.tar')
    simclr = torch.nn.DataParallel(simclr)
    simclr.load_state_dict(dic['state_dict'])
      
    net = ResUNet34_2task(3,simclr).cuda()
    net = torch.load('net.pth')      
    for name,p in net.named_parameters():
        p.requires_grad = True
    
#    frozen_layers = [net.firstconv,net.enc1,net.enc2,net.firstconv_task1,net.enc1_task1,net.enc2_task1]
#    for layer in frozen_layers:
#        for name, p in layer.named_parameters():
#            p.requires_grad = False    
    
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,net.parameters()), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    


    print('----------------' + dataset + '--------------------')


    best_yi_all = 0.
    for epoch in range(max_epoch):
        train_epoch_loss = 0.
        net = net.train()
        print('----------------' + str(epoch) + '--------------------')
        for i, (real_img, label,ves) in enumerate(trainloader):
            
            real_img.requires_grad = False
            label.requires_grad = False
            ves.requires_grad = False
           


            real_img = real_img.cuda()
            label = label.cuda().long()
            #skeleton = skeleton.cuda().float()
            ves = ves.cuda().float()

            optimizer.zero_grad()
            net.zero_grad() 
            fake_2ves, fake_ves = net(real_img)
            loss_ves = lossf_ce(fake_ves, label)
            loss_bves = lossf_bce(fake_2ves,ves)

            #print(loss_ves.item(), loss_bves.item())
            loss = (1-l)*loss_ves + l*loss_bves
            train_epoch_loss = train_epoch_loss + loss
            loss.backward()
            optimizer.step()


        train_epoch_loss = train_epoch_loss / len(trainloader)
        print("loss:",train_epoch_loss.item())
        torch.save(net, 'finalnet.pth')
        #if epoch % 2 == 0: 
        model_acc = test_in_train_AVDRIVE(testloader, net, best_yi_all)
        #    print("model_acc:",model_acc)
        #if model_acc > best_yi_all:
        #    torch.save(net, 'net.pth')
        #    best_yi_all = model_acc
        scheduler.step()

if __name__ == '__main__':
    avdrive_trainset = AVDRIVEloader(dir_avdrive_train_img, dir_avdrive_train_gt,dir_avdrive_test_skeleton, dir_avdrive_train_vessel)
    avdrive_trainloader = torch.utils.data.DataLoader(avdrive_trainset, batch_size=batch_size, shuffle=True)

    avdrive_testset = AVDRIVEloader(dir_avdrive_test_img, dir_avdrive_test_gt, dir_avdrive_test_skeleton, dir_avdrive_test_vessel)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    train_2task_AVDRIVE()
