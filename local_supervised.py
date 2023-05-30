from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
from utils import losses, ramps
from loss.centerloss_onehot import CenterLoss, step_center, step_c_t

args = args_parser()
center_loss_labeled = CenterLoss()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, 30)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        # total local_dataset
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


class SupervisedLocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr

        self.begin_center = False

        self.init_center = True

    def train(self, args, net, op_dict):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        loss_fn = losses.LabelSmoothingCrossEntropy()

        # train and update
        epoch_loss = []
        print('begin sup_training')

        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            print(iter_max)

            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                inputs = image_batch
                ema_inputs = ema_image_batch
                _, features, _, outputs = net(inputs)
                _, _, _, aug_outputs = net(ema_inputs)

                feature = features
                label = label_batch

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

                loss_classification = loss_fn(outputs, label_batch.long()) + loss_fn(aug_outputs, label_batch.long())

                loss = loss_classification

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            with torch.no_grad():
                if self.epoch % 5 == 0:
                    self.center, self.c_t = center_loss_labeled.init_center(self.feature_list, self.label_list)

            with torch.no_grad():
                self.center = step_center(self.center, self.c_t)
                self.c_t = step_c_t(self.center)

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("sup:", epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
