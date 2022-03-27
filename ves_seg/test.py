import sys
sys.path.append("..")
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from DataLoader import * 

dir_avdrive_test_img = "./data/test/images"
dir_avdrive_test_gt = "./data/test/av_label"
dir_avdrive_test_skeleton = "./data/test/skeleton"
dir_avdrive_test_vessel = "./data/test/vessel"


def test_in_train_AVDRIVE(test_loader, net, best_yi_all,epoch):

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
        for i, (test_img, test_label, test_skeleton, test_ves) in enumerate(test_loader):
            test_img, test_label = test_img.cuda(), test_label.cuda()


            test_img_pad = torch.zeros((test_img.shape[0], 3, 640, 640)).cuda()
            test_ves_pad = torch.zeros((test_ves.shape[0], 1, 640, 640)).cuda()
            test_skeleton_pad = torch.zeros((test_skeleton.shape[0], 1, 640, 640)).cuda()

            test_img_pad[:, :, 28:(28+584),37:(37+565)] = test_img
            test_ves_pad[:, :, 28:(28+584),37:(37+565)] = test_ves
            test_skeleton_pad[:,:,28:(28+584),37:(37+565)] = test_skeleton

            net_input = test_img_pad
            label_ves = test_ves_pad
            label_skeleton = test_skeleton_pad
            test_yfake_ves, test_yfake_ske = net(net_input)

        
            test_yfake_ves = test_yfake_ves[:,:,28:(28+584),37:(37+565)]
            test_yfake_ske = test_yfake_ske[:,:,28:(28+584),37:(37+565)]

            label = label[:,28:(28+584),37:(37+565)]

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
            acc_ves = (tp1 + tn1) / (tp1 + tn1 + fp1 + fn1)
            acc_ske = (tp2 + tn2) / (tp2 + tn2 + fp2 + fn2)
            yi_ves = se_ves + sp_ves - 1

            se_all += se_ves
            sp_all += sp_ves
            acc_all += acc_ves
            yi_all += yi_ves
            acc_all_ske += acc_ske

        se_all = se_all / len(test_loader)
        sp_all = sp_all / len(test_loader)
        acc_all = acc_all / len(test_loader)
        yi_all = yi_all / len(test_loader)
        acc_all_ske = acc_all_ske / len(test_loader)

        print('se:', se_all)
        print('sp:', sp_all)
        print('acc:', acc_all)
        print('yi:', yi_all)
        print('acc_ske:', acc_all_ske)
    print('----------------test end-----------------')
    return acc_all

