from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvg
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import DenseNet121
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader
import torch.nn.functional as F


def split(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    print("total local_dataset:", num_items)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    # print(dict_users)
    file = open("split.txt", mode='w')
    file.write(str(dict_users))
    file.close()
    return dict_users


def test(epoch, save_mode_path):
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)
    net = DenseNet121(out_size=7, mode=args.label_uncertainty, drop_rate=0)
    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                           csv_file=args.csv_file_test,
                                           transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize, ]))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
    AUROCs, Accus, Senss, Specs, _, F1 = epochVal_metrics_test(model, test_dataloader, thresh=0.4)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()
    
    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg


def validate(epoch, save_mode_path):
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)
    net = DenseNet121(out_size=7, mode=args.label_uncertainty, drop_rate=0)
    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize, ]))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
    AUROCs, Accus, Senss, Specs, _, F1 = epochVal_metrics_test(model, val_dataloader, thresh=0.4)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()

    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg


# 保存位置
snapshot_path = '/models'

AUROCs = []
Accus = []
Senss = []
Specs = []
F1s = []

supervised_user_id = [0]
unsupervised_user_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
flag_create = False
print('done')

if __name__ == '__main__':

    args = args_parser()
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.basicConfig(filename="log.txt", level=logging.INFO, filemode="w",
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))

    dict_users = split(train_dataset, args.num_users)

    net_glob = DenseNet121(out_size=7, mode=args.label_uncertainty, drop_rate=args.drop_rate)

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[0, 1])

    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, train_dataset, dict_users[i]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr,
                                     betas=(0.9, 0.999), weight_decay=5e-4)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        trainer_locals.append(UnsupervisedLocalUpdate(args, train_dataset, dict_users[i]))

    for com_round in range(args.rounds):
        print("\nbegin com")
        loss_locals = []

        for idx in supervised_user_id:
            if com_round*args.local_ep > 20:
                trainer_locals[idx].base_lr = 3e-4
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            w, loss, op = local.train(args, net_locals[idx], optimizer)
            w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))

        if com_round*args.local_ep > 20:
            if not flag_create:
                print('begin unsup')
                for i in unsupervised_user_id:
                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).cuda())
                    optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr,
                                                 betas=(0.9, 0.999), weight_decay=5e-4)
                    optim_locals.append(copy.deepcopy(optimizer.state_dict()))
                flag_create = True
            for idx in unsupervised_user_id:
                local = trainer_locals[idx]
                optimizer = optim_locals[idx]
                w, loss, op = local.train(args, net_locals[idx], optimizer, com_round*args.local_ep, center_labeled)
                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            center_labeled = trainer_locals[0].center
            for idx in supervised_user_id[1:]:
                center_labeled = center_labeled + trainer_locals[idx].center
            center_labeled = center_labeled / len(supervised_user_id)

        with torch.no_grad():
            w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        for i in supervised_user_id:
            net_locals[i].load_state_dict(w_glob)

        if com_round*args.local_ep > 20:
            for i in unsupervised_user_id:
                net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        # print(loss_avg, com_round)
        logging.info('Loss Avg {}, Round {}, LR {} '.format(loss_avg, com_round, args.base_lr))

        if com_round % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(com_round) + '.pth')
            # torch.save({'state_dict': net_glob.module.state_dict(), }, save_mode_path)
            torch.save({'state_dict': net_glob.state_dict(), }, save_mode_path)

            logging.info("TEST Global: Epoch: {}".format(com_round))

            AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg = test(com_round, save_mode_path)

            AUROC_avg_val, Accus_avg_val, Senss_avg_val, Specs_avg_val, F1_avg_val = validate(com_round, save_mode_path)

            logging.info("TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                         .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))

            logging.info("VAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}, VAL F1: {:6f}"
                         .format(AUROC_avg_val, Accus_avg_val, Senss_avg_val, Specs_avg_val, F1_avg_val))

