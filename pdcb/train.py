import numpy as np
import os
import torchvision.transforms
from model import *
from DataLoader import *
import argparse
from celoss import CrossEntropyLoss
from test import *
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

train_img = "../data/training/images_aug_patch"
train_gt = "../data/training/av_label_aug_patch"
train_gabor = "../DataGray/gabor_aug"
train_vessel = "../data/training/vessel_aug_patch"
train_linear = "../DataGray/linear_aug"
test_img = "../data/test/images"
test_gt = "../data/test/av_all"
test_skeleton = "../data/test/skeleton"
test_vessel = "../data/test/vessel"
test_gabor = "../data/test/gabor"
test_linear = "../data/test/linear"


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

    #j_net = Resnet34_jigsaw(3,340).cuda()
    #j_net = torch.load('./2pre_net.pth')
    net = ResUNet34_2task(3).cuda()
    #net = torch.load('./ffinalnet.pth')

    #ema = EMA(net, 0.999)
    #ema.register()

    #frozen_layers = [net.enc4]
    #for layer in frozen_layers:
    #    for name, p in layer.named_parameters():
    #        p.requires_grad = False
    for name,p in net.named_parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    print('----------------' + dataset + '--------------------')


    best_yi_all = 0.
    for epoch in range(max_epoch):
        train_epoch_loss = 0.
        net = net.train()
        print('----------------' + str(epoch) + '--------------------')
        for i, (real_img, label, ves) in enumerate(trainloader):
            
            real_img.requires_grad = False
            label.requires_grad = False
            ves.requires_grad = False
           


            real_img = real_img.cuda().float()
            label = label.cuda().long()
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
            #ema.update()


        train_epoch_loss = train_epoch_loss / len(trainloader)
        print("loss:",train_epoch_loss.item())
        
        model_acc = test_in_train_AVDRIVE(testloader, net, best_yi_all, epoch)
        #ema.apply_shadow()
        #yi = test_in_train_AVDRIVE(testloader, net, best_yi_all, epoch)
         
        #ema.restore()
        print("model_acc:",model_acc)
        if model_acc > best_yi_all:
            torch.save(net, 'finalnet.pth')
            best_yi_all = model_acc
        scheduler.step()

if __name__ == '__main__':
    avdrive_trainset = AVDRIVEloader(train_img, train_gt, train_vessel, train_gabor, train_linear)
    avdrive_trainloader = torch.utils.data.DataLoader(avdrive_trainset, batch_size=batch_size, shuffle=True)

    avdrive_testset = AVDRIVEloader(test_img, test_gt, test_vessel, test_gabor, test_linear)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    train_2task_AVDRIVE()
