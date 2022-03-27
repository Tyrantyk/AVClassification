import math
import sys
sys.path.append("..")
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from DataLoader import *
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_avdrive_test_img = "../data/test/images"
dir_avdrive_test_gt = "../data/test/av_all"
dir_avdrive_test_skeleton = "../data/test/skeleton"
dir_avdrive_test_vessel = "../data/test/vessel"

# dir_avdrive_test_img = r"D:\CV\CV_Project\avseg\data\training\images_aug_patch"
# dir_avdrive_test_gt = r"D:\CV\CV_Project\avseg\data\training\av_label_aug_patch"
# dir_avdrive_test_skeleton = r"D:\CV\CV_Project\avseg\data\training\skeleton_aug_patch"
# dir_avdrive_test_vessel = r"D:\CV\CV_Project\avseg\data\training\vessel_aug_patch"


def img_show(arr, shape, type="None"):
    if type == 'img':
        show_np = np.zeros((shape, shape, 3))

        red_x, red_y = np.where((arr[0]==1))
        blue_x, blue_y = np.where((arr[0]==2))
        green_x, green_y = np.where((arr[0] == 4))
        purple_x, purple_y = np.where((arr[0] == 3))
        show_np[red_x, red_y, 0] = 255
        show_np[blue_x, blue_y, 2] = 255
        show_np[green_x, green_y, 1] = 255
        show_np[purple_x, purple_y, 0] = 255
        show_np[purple_x, purple_y, 2] = 255
        show_np[purple_x, purple_y, 1] = 255


        show_n = Image.fromarray(show_np.astype('uint8'))
        show_n.show()
        # show_np.save(r"D:\CV\CV_Project\avseg\data\test\predict" + str(i) + 'pre.jpg')

    if type == 'ves':
        show_np = np.zeros((shape, shape,3))
        ves_x, ves_y = np.where((arr == 1))
        wrong_x, wrong_y = np.where((arr == 2))
        miss_x, miss_y = np.where((arr == 3))
        show_np[ves_x, ves_y, 0] = 255
        show_np[ves_x, ves_y, 1] = 255
        show_np[ves_x, ves_y, 2] = 255
        show_np[wrong_x, wrong_y, 1] = 255
        show_np[miss_x, miss_y, 2] = 255
        show_np = Image.fromarray(show_np.astype('uint8'))
        show_np.show()
        # show_np.save(r"D:\CV\CV_Project\avseg\data\test\predict" + str(i) + 'pre.jpg')

    if type == 'gt':
        show_np = np.zeros((shape, shape, 3))
        red_x, red_y = np.where((arr[0] == 1))
        blue_x, blue_y = np.where((arr[0] == 2))
        show_np[red_x, red_y, 0] = 255
        show_np[blue_x, blue_y, 2] = 255
        show_np = Image.fromarray(show_np.astype('uint8'))
        show_np.show()
        # show_np.save(r"D:\CV\CV_Project\avseg\data\test\predict" + str(i) + 'pre.jpg')

def ves_seg_eval(pre, label, factor):
    pos_cnt = 0
    neg_cnt = 0
    gt_cnt = 0
    ves_pre_gt = np.zeros((pre.shape[2], pre.shape[3]))

    for i in range(len(pre[0][0])):
        for j in range(len(pre[0][0][0])):
            # print(ves_seg[0][0][t][u].item(),end=" ")
            if pre[0][0][i][j].item() > factor:  # positive samples
                if label[0][0][i][j]:
                    pos_cnt += 1
                    ves_pre_gt[i][j] = 1
                else:
                    neg_cnt += 1
                    ves_pre_gt[i][j] = 2
            elif pre[0][0][i][j].item() < factor and label[0][0][i][j]:
                ves_pre_gt[i][j] = 3
            # print(label[0][0][i][j])
            if label[0][0][i][j]:
                gt_cnt += 1

    ves_acc = pos_cnt / gt_cnt
    wrong_rate = neg_cnt / gt_cnt
    return ves_acc, wrong_rate, ves_pre_gt

def av_cla_eval(pre, label, ves):
    VesBased_pre = np.zeros(label.shape)
    pre_dif_show = np.zeros(label.shape)
    for i in range(len(label[0])):
        for j in range(len(label[0][0])):
            if ves[i][j] == 1 or ves[i][j] == 2:
                VesBased_pre[0][i][j] = 1 if pre[0][1][i][j] > pre[0][2][i][j] else  2
            if label[0][i][j] == 1 and VesBased_pre[0][i][j] == 2:
                pre_dif_show[0][i][j] = 1
            elif label[0][i][j] == 2 and VesBased_pre[0][i][j] == 1:
                pre_dif_show[0][i][j] = 2
            elif label[0][i][j] != 0 and VesBased_pre[0][i][j] == 0:
                pre_dif_show[0][i][j] = 3
            elif label[0][i][j] != 0:
                pre_dif_show[0][i][j] = 4

    label = label.reshape(-1)
    VesBased_pre = VesBased_pre.reshape(-1)

    matrix = confusion_matrix(label, VesBased_pre)

    tp, fn, fp, tn = matrix[1, 1], matrix[1, 2], matrix[2, 1], matrix[2, 2]

    new_eval = (tp + tn) / (tp + tn + fp + fn + matrix[1, 0] + matrix[2, 0])
    old_eval = (tp + tn) / (tp + tn + fp + fn)
    return old_eval, new_eval, pre_dif_show

def test_in_train_AVDRIVE(test_loader, net, best_yi_all):

    print('----------------test begin-----------------')
    with torch.no_grad():


        ves_acc_all = 0.
        wrong_rate_all = 0.
        new_acc_all = 0.
        old_acc_all = 0.

        for i, (test_img, test_label, test_ves) in enumerate(test_loader):
            test_img, test_label, test_ves = test_img.cuda(), test_label.cuda(), test_ves.cuda()
            net_input = test_img
            label = test_label

            test_img_pad = torch.zeros((test_img.shape[0], 3, 640, 640)).cuda()
            test_label_pad = torch.zeros((test_label.shape[0], 640, 640)).cuda()
            test_ves_pad = torch.zeros((1, 1, 640, 640)).cuda()
            test_img_pad[:,:,28:(28+584),37:(37+565)] = test_img
            test_label_pad[:, 28:(28+584),37:(37+565)] = test_label
            test_ves_pad[:,:,28:(28+584),37:(37+565)] = test_ves
            net_input = test_img_pad
            label = test_label_pad
            ves_seg, test_yfake_ves = net(net_input)


            label = label.detach().cpu().numpy()
            test_yfake_ves = test_yfake_ves.detach().cpu().numpy()

            ves_acc, wrong_rate, ves_pre_gt = ves_seg_eval(ves_seg, test_ves_pad, 0.1)
            old_eval, new_eval, pre_dif_show = av_cla_eval(test_yfake_ves, label, ves_pre_gt)

            #img_show(pre_dif_show,pre_dif_show.shape[1],'img')
            #img_show(ves_pre_gt, ves_pre_gt.shape[0], 'ves')

            print('ves_acc:',ves_acc, 'wrong_rate:',wrong_rate, 'old:',old_eval, 'new:',new_eval)
            ves_acc_all += ves_acc
            wrong_rate_all += wrong_rate
            new_acc_all += new_eval
            old_acc_all += old_eval

        new_acc_all = new_acc_all / len(test_loader)
        old_acc_all = old_acc_all / len(test_loader)
        ves_acc_all = ves_acc_all / len(test_loader)
        wrong_rate_all = wrong_rate_all / len(test_loader)

        print('old_acc:',old_acc_all)
        print('new_acc:', new_acc_all)
        print('ves_acc:', ves_acc_all)
        print('wrong_rate:', wrong_rate_all)

    print('----------------test end-----------------')
    return new_acc_all





if __name__ == '__main__':
    avdrive_testset = AVDRIVEloader(dir_avdrive_test_img, dir_avdrive_test_gt, dir_avdrive_test_skeleton,dir_avdrive_test_vessel)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    net = ResUNet34_2task(3).cuda()
    # model_dict = torch.load('./finalnet.pth',map_location=torch.device('cpu')).module.state_dict()
    # net.load_state_dict(model_dict)
    # net = torch.nn.DataParallel(net)
    net = torch.load('net.pth')

    yi = test_in_train_AVDRIVE(avdrive_testloader,net,0)
