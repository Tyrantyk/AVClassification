import sys
sys.path.append("..")
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from DataLoader import * 
from model import *
dir_avdrive_test_img = "/workspace/workspace/data/test/images"
dir_avdrive_test_gt = "/workspace/workspace/data/test/av_label"
dir_avdrive_test_skeleton = "/workspace/workspace/data/test/skeleton"
dir_avdrive_test_vessel = "/workspace/workspace/data/test/vessel"
dir_avdrive_test_mask = "/workspace/workspace/data/test/mask"

def test_in_train_AVDRIVE(test_loader, net, best_yi_all):

    print('----------------test begin-----------------')
    with torch.no_grad():
        test_labels = np.array([])
        test_probs = np.array([])

        test_loss = 0.

        se_all = 0.
        sp_all = 0.
        acc_all = 0.
        yi_all = 0.
        acc_all_ske = 0.
        acc_gt_all = 0.
        for i, (test_img, test_label, test_skeleton, test_ves, test_mask) in enumerate(test_loader):
            test_img, test_label, test_skeleton, test_ves, test_mask = test_img.cuda(), test_label.cuda(), test_skeleton.cuda(), test_ves.cuda(), test_mask.cuda()


            test_img_pad = torch.zeros((test_img.shape[0], 3, 640, 640)).cuda()
            test_ves_pad = torch.zeros((test_ves.shape[0], 1, 640, 640)).cuda()
            test_mask_pad = torch.zeros((test_mask.shape[0], 1, 640, 640)).cuda()
            test_skeleton_pad = torch.zeros((test_skeleton.shape[0], 1, 640, 640)).cuda()
           
            test_img_pad[:, :, 28:(28+584),37:(37+565)] = test_img
            test_ves_pad[:, :, 28:(28+584),37:(37+565)] = test_ves
            test_skeleton_pad[:,:,28:(28+584),37:(37+565)] = test_skeleton
            test_mask_pad[:, :, 28:(28+584),37:(37+565)] = test_mask            
            mask = test_mask_pad.detach().cpu().numpy()
            test_ves = test_ves.detach().cpu().numpy()
            mask_cnt = mask[mask==1].size
            mask_drop  = mask[mask==0].size
            gt_cnt = test_ves[test_ves==1].size
            #print(mask, type(mask))
            net_input = test_img_pad
            label_ves = test_ves_pad
            label_skeleton = test_skeleton_pad
            
            test_yfake_ves, test_yfake_ske = net(net_input)

            # test_yfake_ves = torch.nn.functional.softmax(test_yfake_ves, dim=1)
            # test_yfake_ves = torch.argmax(test_yfake_ves, dim=1)
            test_yfake_ves = torch.where(test_yfake_ves > 0.5, 1, 0)
            test_yfake_ske = torch.where(test_yfake_ske > 0.5, 1, 0)

            # label = label.detach().cpu().numpy()
            label_ves = label_ves.detach().cpu().numpy()
            label_skeleton = label_skeleton.detach().cpu().numpy()

            test_yfake_ves = test_yfake_ves.squeeze().detach().cpu().numpy()
            test_yfake_ske = test_yfake_ske.squeeze().detach().cpu().numpy()

            label_ves = label_ves.reshape(-1)
            label_skeleton = label_skeleton.reshape(-1)
            test_yfake_ves = test_yfake_ves.reshape(-1)
            test_yfake_ske = test_yfake_ske.reshape(-1)

            matrix_ves = confusion_matrix(label_ves, test_yfake_ves)
            matrix_ske = confusion_matrix(label_skeleton, test_yfake_ske)
            tp1, fn1, fp1, tn1= matrix_ves[0,0], matrix_ves[0,1], matrix_ves[1,0], matrix_ves[1,1]
            tp2, fn2, fp2, tn2 = matrix_ske[0, 0], matrix_ske[0, 1], matrix_ske[1, 0], matrix_ske[1, 1]

            se_ves = tp1 / (tp1+fn1)
            sp_ves = tn1 / (tn1+fp1)
            acc_ves = (tp1 + tn1 - mask_drop) / mask_cnt
            acc_ske = (tp2 + tn2 - mask_drop) / mask_cnt
            acc_gt = tn1 / gt_cnt
            yi_ves = se_ves + sp_ves - 1
            print(acc_gt, acc_ves)
            se_all += se_ves
            sp_all += sp_ves
            acc_all += acc_ves
            acc_gt_all += acc_gt
            yi_all += yi_ves
            acc_all_ske += acc_ske

        se_all = se_all / len(test_loader)
        sp_all = sp_all / len(test_loader)
        acc_all = acc_all / len(test_loader)
        yi_all = yi_all / len(test_loader)
        acc_gt_all = acc_gt_all / len(test_loader)
        acc_all_ske = acc_all_ske / len(test_loader)

        print('se:', se_all)
        print('sp:', sp_all)
        print('acc:', acc_all)
        print('acc_gt:', acc_gt_all)
        print('yi:', yi_all)
        print('acc_ske:', acc_all_ske)
    print('----------------test end-----------------')
    return acc_all

if __name__ == '__main__':
    avdrive_testset = AVDRIVEloader(dir_avdrive_test_img, dir_avdrive_test_gt, dir_avdrive_test_skeleton,dir_avdrive_test_vessel, dir_avdrive_test_mask)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    net = ResUNet34_2task(3)
    # # model_dict = torch.load('./finalnet.pth',map_location=torch.device('cpu')).module.state_dict()
    # # net.load_state_dict(model_dict)
    # # net = torch.nn.DataParallel(net)
    net = torch.load('net.pth')
    #
    yi = test_in_train_AVDRIVE(avdrive_testloader,net,0)
