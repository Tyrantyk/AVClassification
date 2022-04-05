import numpy as np
import os
import torchvision.transforms
from model import *
from DataLoader import *
import argparse
from networks.TransSegNet_81 import TransSegNet_81
from test import *
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dir_avdrive_train_img = "/workspace/workspace/data/training/images_aug_patch"
dir_avdrive_train_gt = "/workspace/workspace/data/training/av_label_aug_patch"
dir_avdrive_train_skeleton = "/workspace/workspace/data/training/skeleton_aug_patch"
dir_avdrive_train_vessel = "/workspace/workspace/data/training/vessel_aug_patch"
dir_avdrive_test_img = "/workspace/workspace/data/test/images"
dir_avdrive_test_gt = "/workspace/workspace/data/test/av_all"
dir_avdrive_test_skeleton = "/workspace/workspace/data/test/skeleton"
dir_avdrive_test_vessel = "/workspace/workspace/data/test/vessel"
dir_avdrive_test_mask = "/workspace/workspace/data/test/mask"

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

def train_2task_AVDRIVE():


    trainloader = avdrive_trainloader
    testloader = avdrive_testloader

    net = TransSegNet_81().cuda()


    for name,p in net.named_parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    

    print('----------------AVDrive--------------------')

    best_yi_all = 0.
    for epoch in range(max_epoch):
        train_epoch_loss = 0.
        net = net.train()
        print('----------------' + str(epoch) + '--------------------')
        for i, (real_img, label,skeleton, ves) in enumerate(trainloader):
            
            real_img.requires_grad = False
            label.requires_grad = False
            skeleton.requires_grad  = False
            ves.requires_grad = False

            real_img = real_img.cuda()
            label = label.cuda().long()
            skeleton = skeleton.cuda().float()
            ves = ves.cuda().float()

            optimizer.zero_grad()
            net.zero_grad()
            
            # fake_2ves, fake_ves = net(real_img)
            # loss_ves = lossf_ce(fake_ves, label)
            #pre_2ves, pre_skelton= net(real_img)
            pre_2ves , pre2= net(real_img)
            loss_ves1 = lossf_bce(pre_2ves,ves)
            loss2 = lossf_bce(pre2,ves)
            #loss_skelton = lossf_bce(pre_skelton,skeleton)
            #print(loss_ves.item(), loss_bves.item())
            #loss = (1-l)*loss_ves + l*loss_skelton
            loss = loss_ves1 + loss2
            train_epoch_loss = train_epoch_loss + loss
            loss.backward()
            optimizer.step()


        train_epoch_loss = train_epoch_loss / len(trainloader)
        print("loss:",train_epoch_loss.item())
        
        model_acc = test_in_train_AVDRIVE(testloader, net, best_yi_all)
        #torch.save(net, 'net.pth')
        print("model_acc:",model_acc)
        if model_acc > best_yi_all:
            torch.save(net, 'net.pth')
            best_yi_all = model_acc
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

if __name__ == '__main__':
    avdrive_trainset = AVDRIVEloader(dir_avdrive_train_img, dir_avdrive_train_gt,dir_avdrive_train_skeleton, dir_avdrive_train_vessel)
    avdrive_trainloader = torch.utils.data.DataLoader(avdrive_trainset, batch_size=batch_size, shuffle=True)

    avdrive_testset = AVDRIVEloader(dir_avdrive_test_img, dir_avdrive_test_gt, dir_avdrive_test_skeleton, dir_avdrive_test_vessel, dir_avdrive_test_mask)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    train_2task_AVDRIVE()
