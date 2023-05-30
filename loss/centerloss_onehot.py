import torch
import numpy as np
import torch.nn.functional as F

CLASS_NUM = 7


class CenterLoss:
    def __init__(self):
        super(CenterLoss, self).__init__()

    def init_center(self, feature, label):
        one_hot_label = np.eye(CLASS_NUM)
        one_hot_label = torch.tensor(one_hot_label).int().cuda()
        global sum_feature, sum_index
        for i in range(CLASS_NUM):
            index = torch.eq(label, one_hot_label[i]).int()
            index = index.sum(axis=1, keepdim=True)
            index = torch.eq(index, CLASS_NUM).int()
            index_i = index.sum()
            feature_i = feature.mul(index)
            if i == 0:
                sum_feature = torch.sum(feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(index_i).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(feature_i, dim=0).view(1, feature.shape[1])], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(index_i).view(1, 1)], dim=0)
        sum_index = sum_index + 1
        center = sum_feature / sum_index
        c_t = torch.sum(center, dim=0) / CLASS_NUM
        return center, c_t

    def cal_center(self, feature, label, center):
        one_hot_label = np.eye(CLASS_NUM)
        one_hot_label = torch.tensor(one_hot_label).int().cuda()
        global f_c
        for j in range(CLASS_NUM):
            index = torch.eq(label, one_hot_label[j]).int()
            index = index.sum(axis=1, keepdim=True)
            index = torch.eq(index, CLASS_NUM).int()
            f_c_0 = feature.mul(index) - center[j]
            if j == 0:
                f_c = torch.norm(f_c_0.mul(index), p=2, dim=1)
            else:
                f_c = f_c + torch.norm(f_c_0.mul(index), p=2, dim=1)
        center_loss = f_c.sum() / feature.shape[0]
        return center_loss


def step_center(center, c_t):
    alpha = 0.002
    step_center = center + alpha * ((center-c_t) / torch.norm(center-c_t, p=2, dim=1).view((CLASS_NUM, -1)))
    return step_center


def step_c_t(center):
    step_c_t = torch.sum(center, dim=0) / CLASS_NUM
    return step_c_t


def co_center(center_unlabeled, center_labeled):
    center_unlabeled = F.normalize(center_unlabeled, dim=1)
    center_labeled = F.normalize(center_labeled, dim=1)
    center_consistency_loss = F.mse_loss(center_unlabeled, center_labeled, reduction='sum') / CLASS_NUM
    return center_consistency_loss

