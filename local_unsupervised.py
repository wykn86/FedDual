from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import DenseNet121
from utils import losses, ramps
from loss.centerloss_onehot import CenterLoss, step_center, step_c_t, co_center
from loss.pgcloss_label import PseudoGroupContrast

args = args_parser()
center_loss_unlabeled = CenterLoss()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, 30)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)

        net = DenseNet121(out_size=7, mode=args.label_uncertainty, drop_rate=args.drop_rate)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])

        self.ema_model = net.cuda()
        for param in self.ema_model.parameters():
            param.detach_()

        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4#args.base_lr

        self.contrast_loss_unlabeled = PseudoGroupContrast()

        self.begin_center = False

        self.init_center = True

    def train(self, args, net, op_dict, epoch, center_labeled):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        # init center
        if self.epoch == 0:
            self.center = center_labeled
            self.c_t = torch.sum(center_labeled, dim=0) / center_labeled.size(0)

        self.epoch = epoch

        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False
            print('done')

        epoch_loss = []
        print('begin unsup_training')

        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            print(iter_max)

            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()

                inputs = image_batch
                ema_inputs = ema_image_batch
                _, features, projectors, outputs = net(inputs)
                with torch.no_grad():
                    _, _, ema_projectors, ema_outputs = self.ema_model(ema_inputs)

                feature = features
                pseudo_label_batch = F.softmax(ema_outputs, dim=1)
                pseudo_label_batch = torch.argmax(pseudo_label_batch, dim=1)
                pseudo_label_batch = F.one_hot(pseudo_label_batch, num_classes=7)
                label = pseudo_label_batch

                if self.init_center is True:
                    feature_list = feature
                    label_list = label
                else:
                    feature_list = torch.cat([feature, self.feature_list], dim=0)
                    label_list = torch.cat([label, self.label_list], dim=0)
                    length_max = 701
                    if len(feature_list) > length_max:
                        feature_list = feature_list[0:length_max, :]
                        label_list = label_list[0:length_max, :]
                self.init_center = False

                self.feature_list = feature_list.clone().detach()
                self.label_list = label_list.clone().detach()

                center_batch, _ = center_loss_unlabeled.init_center(feature, label)
                co_center_loss = co_center(center_batch, center_labeled)
                # print("co-center-loss:", co_center_loss)

                contrast_loss = self.contrast_loss_unlabeled.forward(projectors, ema_projectors, label)
                # print("contrast-loss:", contrast_loss)

                consistency_weight = get_current_consistency_weight(self.epoch)
                consistency_loss = torch.sum(losses.softmax_mse_loss(outputs, ema_outputs)) / args.batch_size

                loss = 15 * consistency_weight * consistency_loss + 5 * consistency_weight * co_center_loss

                if self.epoch > 20:
                    loss = loss + 0.01 * consistency_weight * contrast_loss

                loss = 1.0 * loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            with torch.no_grad():
                if self.epoch % 5 == 0:
                    self.center, self.c_t = center_loss_unlabeled.init_center(self.feature_list, self.label_list)

            with torch.no_grad():
                self.center = step_center(self.center, self.c_t)
                self.c_t = step_c_t(self.center)

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("unsup:", epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
