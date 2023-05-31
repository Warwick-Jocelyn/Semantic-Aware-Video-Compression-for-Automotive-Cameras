#import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import SimpleITK as sitk

device = torch.device("cuda:0")
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        # cc_size = cc.shape
        # cc_sum = torch.sum(cc)
        # target_value = 0.99
        # target_sum = torch.sum(cc[cc>target_value])
        # target_size = cc[cc>target_value].shape
        # print(cc_size[0]*cc_size[1]*cc_size[2]*cc_size[3]*cc_size[4])
        # print(target_size[0])

        # mean = (cc_sum-target_sum)/(cc_size[0]*cc_size[1]*cc_size[2]*cc_size[3]*cc_size[4]-target_size[0])

        # print(cc_size)
        # return -mean

        #print(torch.sum(cc[cc>0.99]))


        return -torch.mean(cc)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self,  y_pred):
        dy = torch.abs(y_pred[:, :, :, 1:, :, :] - y_pred[:, :, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, :, 1:, :] - y_pred[:, :, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, :, 1:] - y_pred[:, :, :, :, :, :-1])
        dt = torch.abs(y_pred[:, :, 1:, :, :, :] - y_pred[:, :, :-1, :, :, :])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
            dt = dt * dt

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz) + torch.mean(dt)
        grad = d / 4.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)
        dice_sum = 0

        x = 0
        y = 0

        # inter = torch.sum(logits[:, 0, :, :, :] * targets[:, 0, :, :, :])
        # union = torch.sum(logits[:, 0, :, :, :]) + torch.sum(targets[:, 0, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # dice_sum += 0*dice
        # ###rv
        # inter = torch.sum(logits[:, 1, :, :, :] * targets[:, 1, :, :, :])
        # union = torch.sum(logits[:, 1, :, :, :]) + torch.sum(targets[:, 1, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # # print('1:',dice)
        #
        # # if dice > 0.97:
        # #     x = 0
        # # else:
        # #     x = 0.25
        # x = 0.3
        # dice_sum += x * dice
        # ##LV
        # inter = torch.sum(logits[:, 3, :, :, :] * targets[:, 3, :, :, :])
        # union = torch.sum(logits[:, 3, :, :, :]) + torch.sum(targets[:, 3, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # # print('3:', dice)
        # if dice > 0.97:
        #     y = 0
        # else:
        #     y = 0.25
        #
        # dice_sum += y * dice
        # ##myo
        # inter = torch.sum(logits[:, 2, :, :, :] * targets[:, 2, :, :, :])
        # union = torch.sum(logits[:, 2, :, :, :]) + torch.sum(targets[:, 2, :, :, :])
        # dice = (2. * inter + 1) / (union + 1)
        # # print('2:', dice)
        #
        # dice_sum += (1-x-y) * dice
        #
        # return 1-dice_sum

        for i in range(class_num):
            if i==0:
                inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
                union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
                dice = (2. * inter + 1) / (union + 1)
                dice_sum += dice*0.1
            elif i==1:
                inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
                union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
                dice = (2. * inter + 1) / (union + 1)
                dice_sum += dice * 0.35
            elif i == 2:
                inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
                union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
                dice = (2. * inter + 1) / (union + 1)
                dice_sum += dice * 0.35
            else:
                inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
                union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
                dice = (2. * inter + 1) / (union + 1)
                dice_sum += dice * 0.2


            # inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            # union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            # dice = (2. * inter + 1) / (union + 1)
            # dice_sum += dice
            # print(i,':',dice)
        # return 1 - dice_sum / class_num

        return 1 - dice_sum

class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):

        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0]).to(device)
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)

        return 1 - 2. * inter / union

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice

from torch.autograd import Variable
import random
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def location(seges,seged,bias,category):
    one = torch.ones_like(seges)
    zero = torch.zeros_like(seges)
    mi3=0
    mi4=0
    mx3=0
    mx4=0

    size = seges.shape

    if category ==4:
        seg_es1 = torch.unsqueeze(
            torch.unsqueeze(seges[0, 1, :, :, :] * 1 + seges[0, 2, :, :, :] * 2 + seges[0, 3, :, :, :] * 3, 0), 0)
        seg_es1 = seg_es1[:, :, :, :, :]
        one = torch.ones_like(seg_es1)
        zero = torch.zeros_like(seg_es1)



        seg_ed1 = torch.unsqueeze(
            torch.unsqueeze(seged[0, 1, :, :, :] * 1 + seged[0, 2, :, :, :] * 2 + seged[0, 3, :, :, :] * 3, 0), 0)

        seg_ed1 = seg_ed1[:, :, :, :, :]


    else:
        seg_es1 = torch.unsqueeze(
            torch.unsqueeze(seged[0, category, :, :, :] * category , 0), 0)
        seg_ed1 = torch.unsqueeze(
            torch.unsqueeze(seged[0, category, :, :, :] * category , 0), 0)

    seg_es1 = torch.where(seg_es1 >= 1, one, zero)
    seg_es1_0 = np.nonzero(seg_es1)
    seg_es_row3 = seg_es1_0[:, 3]
    mx_seg_es_row3 = torch.max(seg_es_row3)
    mi_seg_es_row3 = torch.min(seg_es_row3)
    seg_es_row4 = seg_es1_0[:, 4]
    mx_seg_es_row4 = torch.max(seg_es_row4)
    mi_seg_es_row4 = torch.min(seg_es_row4)

    seg_ed1 = torch.where(seg_ed1 >= 1, one, zero)
    seg_ed1_0 = np.nonzero(seg_ed1)
    seg_ed_row3 = seg_ed1_0[:, 3]
    mx_seg_ed_row3 = torch.max(seg_ed_row3)
    mi_seg_ed_row3 = torch.min(seg_ed_row3)
    seg_ed_row4 = seg_ed1_0[:, 4]
    mx_seg_ed_row4 = torch.max(seg_ed_row4)
    mi_seg_ed_row4 = torch.min(seg_ed_row4)

    if mi_seg_es_row3<mi_seg_ed_row3:
        mi3 = mi_seg_es_row3
    else:
        mi3 = mi_seg_ed_row3

    if mi_seg_es_row4<mi_seg_ed_row4:
        mi4 = mi_seg_es_row4
    else:
        mi4 = mi_seg_ed_row4

    if mx_seg_es_row3<mx_seg_ed_row3:
        mx3 = mx_seg_ed_row3
    else:
        mx3 = mx_seg_es_row3

    if mx_seg_es_row4<mx_seg_ed_row4:
        mx4 = mx_seg_ed_row4
    else:
        mx4 = mx_seg_es_row4


    mi3 = mi3-bias
    if mi3<0:
        mi3 = 0

    mi4 = mi4-bias
    if mi4<0:
        mi4 = 0

    if mx3+bias>size[3]:
        mx3 = size[3]
    else:
        mx3 = mx3+bias

    if mx4+bias>size[4]:
        mx4 = size[4]
    else:
        mx4 = mx4+bias

    return mi3,mx3,mi4,mx4

class crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth))

class DiceMeanLoss1(nn.Module):
    def __init__(self):
        super(DiceMeanLoss1, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)
        dice_sum = 0

        x = 0
        y = 0


        for i in range(class_num):


            inter = torch.sum(logits[:, i,  :, :] * targets[:, i,  :, :])
            union = torch.sum(logits[:, i,  :, :]) + torch.sum(targets[:, i,  :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
            # print(i,':',dice)
        return 1 - dice_sum / class_num, inter/union - inter
        # return 1 - dice_sum

class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)
        iou_sum = 0

        x = 0
        y = 0

        for i in range(class_num):
            inter = torch.sum(logits[:, i,  :, :] * targets[:, i,  :, :])
            union_with_overlap = torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :])
            union = union_with_overlap - inter
            iou = inter/union
            iou_sum += iou
            # print(i,':',iou)
        return iou_sum / class_num
