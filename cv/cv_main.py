import argparse
import random
import sys
sys.path.append("../multi_graph")
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import time
import warnings
warnings.filterwarnings("ignore")

from albumentations import (
    Resize, ShiftScaleRotate, Flip, Compose, Normalize, RandomBrightnessContrast
)
from albumentations.pytorch import ToTensorV2

from utils import AverageMeter, adjust_learning_rate, balance_five_weights, save_checkpoint, _ids2imgpaths, _img2ids

from graph_input import MultiTaskCalsDataset, MultiTaskCalsListDataset, _read_txt
from mg_conv import genconv_mg
from torch.utils.data.sampler import WeightedRandomSampler

import torch_geometric.transforms as T

import wandb
# wandb.init(project="multi-task")

# Fix the too many open files bug
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark=True

# Set random seed.
def set_random_seed(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser(description='PyTorch Graph Training')

parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int,
                    metavar='N', help="number of total epochs to run")
parser.add_argument('--num_graph_classes', dest='graph_num_classes', default=5, type=int)
parser.add_argument('--num_node_classes', dest='node_num_classes', default=5, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int,
                    metavar='N')
parser.add_argument('--num_features', dest='num_features', default=66, type=int)
parser.add_argument('--hidden_size', dest='hidden_size', default=128, type=int,
                    metavar='N')
parser.add_argument('--learning_rate', dest='learning_rate', default=3e-4,
                    type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--num_workers', dest='num_workers', default=8, type=int,
                    metavar='N', help="number of workers for dataloader")
parser.add_argument('--momentum', dest='momentum', default=0.9, type=float,
                    metavar='M', help="momentum")
parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float,
                    metavar='W', help="weight decay (default: 1e-4)")
parser.add_argument('--pretrain', default=0, type=int, metavar='PATH',
                    help='whether initialize with pretrain model')
parser.add_argument('--pretrain_path', metavar='pretrain_path',
                    default='/space/tmu_graph_models/patch_model/model_best.pth.tar', type=str)
parser.add_argument('--model_path', metavar='model_path',
                    default='../models/cv_mg/checkpoint.pth.tar', type=str)
parser.add_argument('--restart', default=1, type=int, metavar='RESTART', help='whether restart training')
parser.add_argument('--patch_size', default=14, type=int, metavar='patch_size')
parser.add_argument('--train_txt', metavar='train_txt',
                    default='../data_list/train_multilabel_v3.txt', type=str)
parser.add_argument('--test_txt', metavar='test_txt',
                    default='../data_list/test_multilabel_v3.txt', type=str)
parser.add_argument('--num_layers', default=8, type=int)
parser.add_argument('--alpha', default=1.5, type=float)

args = parser.parse_args()

distribution_classes = ["cluster", "linear", "segmental", "regional", "diffuse"]
morphology_classes = ['not suspicious', 'amorphous', 'heterogeneous', 'pleomorphic', 'linear']
tasks = ['node', 'graph']
initial_task_loss = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Weight(torch.nn.Module):
    def __init__(self, tasks):
        super(Weight, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

def train(train_loader, model, criterion1, criterion2, optimizer, epoch, total_step, Weights, initial_task_loss, graph_transform=None):
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    for i, (patch_tensor, coord_tensor, morph_tensor, dist_label) in enumerate(train_loader):

        patch_tensor = patch_tensor.view(-1, args.patch_size * args.patch_size).to(device)
        coord_tensor = coord_tensor.squeeze().to(device)

        node_pred, graph_pred = model(patch_tensor, coord_tensor)

        morph_tensor = morph_tensor.squeeze().to(device)
        dist_label = dist_label.to(device)

        loss1 = criterion1(node_pred, morph_tensor)
        loss2 = criterion2(graph_pred, dist_label)

        norms = []
        w = torch.zeros(len(tasks)).float().to(device)
        W = model.layers[0].conv.mlp[-1].weight
        train_loss = [loss1, loss2]

        for j, t in enumerate(tasks):
            gygw = torch.autograd.grad(train_loss[j], W, retain_graph=True)
            norms.append(torch.norm(torch.mul(Weights.weights[j], gygw[0])))

        norms = torch.stack(norms)
        task_loss = torch.stack(train_loss)

        if epoch == 0 and i == 0:
            initial_task_loss = task_loss

        loss_ratio = task_loss.data / initial_task_loss.data
        inverse_train_rate = loss_ratio / loss_ratio.mean()

        mean_norm = norms.mean()
        constant_term = mean_norm.data * (inverse_train_rate ** args.alpha)
        grad_norm_loss = (norms - constant_term).abs().sum()

        w_grad = torch.autograd.grad(grad_norm_loss, Weights.weights)[0]
        for j in range(len(tasks)):
            w[j] = Weights.weights[j].data

        loss = sum(w[j].data * train_loss[j] for j in range(len(tasks)))

        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad() # Clear gradients.
        loss.backward()  # Derive gradients.

        Weights.weights.grad = torch.zeros_like(Weights.weights.data)
        Weights.weights.grad.data = w_grad.data

        optimizer.step()  # Update parameters based on gradients.

        Weights.weights.data = len(tasks) * Weights.weights.data / Weights.weights.data.sum()

        if (i+1) % 50 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch+1, args.num_epochs, i+1, total_step, losses.avg
            ))

    return losses.avg, Weights, initial_task_loss

def validate(test_loader, model, criterion1, criterion2, Weights, graph_transform=None):
    val_losses = AverageMeter()
    val_node_losses = AverageMeter()
    val_graph_losses = AverageMeter()

    # Test the model
    model.eval()  # eval model (batchnorm uses moving mean/variance instead of mini-batch

    with torch.no_grad():
        node_correct = 0
        graph_correct = 0
        node_total = 0
        graph_total = 0

        node_labels = []
        graph_labels = []
        node_scores = []
        graph_scores = []
        node_preds = []
        graph_preds = []

        for patch_tensor, coord_tensor, morph_tensor, dist_label in test_loader:
            patch_tensor = patch_tensor.view(-1, args.patch_size * args.patch_size).to(device)
            coord_tensor = coord_tensor.squeeze().to(device)

            morph_tensor = morph_tensor.squeeze().to(device)
            dist_label = dist_label.to(device)

            node_logits, graph_logits = model(patch_tensor, coord_tensor)

            val_loss1 = criterion1(node_logits, morph_tensor)
            val_loss2 = criterion2(graph_logits, dist_label)
            val_loss = val_loss1 + val_loss2
            val_losses.update(val_loss.item(), args.batch_size)
            val_node_losses.update(val_loss1.item(), args.batch_size)
            val_graph_losses.update(val_loss2.item(), args.batch_size)

            node_labels += list(morph_tensor.cpu().numpy())
            graph_labels += list(dist_label.cpu().numpy())

            node_pred = node_logits.argmax(dim=1)
            node_preds += list(node_pred.cpu().numpy())
            graph_pred = graph_logits.argmax(dim=1)
            graph_preds += list(graph_pred.cpu().numpy())

            node_score = F.softmax(node_logits, dim=1).cpu().data.numpy()
            graph_score = F.softmax(graph_logits, dim=1).cpu().data.numpy()

            node_total += morph_tensor.size(0)
            graph_total += dist_label.size(0)
            node_correct += (node_pred==morph_tensor).sum().item()
            graph_correct += (graph_pred==dist_label).sum().item()

            node_acc = 100*node_correct/node_total
            graph_acc = 100*graph_correct/graph_total

            node_scores.append(node_score)
            graph_scores.append(graph_score)

        final_node_scores = np.vstack(node_scores)
        final_graph_scores = np.vstack(graph_scores)

        node_auc = roc_auc_score(node_labels, final_node_scores, multi_class='ovr')
        graph_auc = roc_auc_score(graph_labels, final_graph_scores, multi_class='ovr')

        print('Test Accuracy of node: {}%'.format(100 * node_correct / node_total))
        print('Test AUC of node: {}'.format(node_auc))
        print('Test Accuracy of graph: {}%'.format(100 * graph_correct / graph_total))
        print('Test AUC of graph: {}'.format(graph_auc))

        node_roc_auc = dict()
        node_y = label_binarize(node_labels, classes=[0, 1, 2, 3, 4])
        for i in range(5):
            node_roc_auc[i] = roc_auc_score(node_y[:, i], final_node_scores[:, i])
            print("AUC of class {0} (area = {1:0.2f})".format(morphology_classes[i], node_roc_auc[i]))
        print(Counter(node_preds).items())
        print(Counter(node_labels).items())

        graph_roc_auc = dict()
        graph_y = label_binarize(graph_labels, classes=[0, 1, 2, 3, 4])
        for i in range(5):
            graph_roc_auc[i] = roc_auc_score(graph_y[:, i], final_graph_scores[:, i])
            print("AUC of class {0} (area = {1:0.2f})".format(distribution_classes[i], graph_roc_auc[i]))
        print(Counter(graph_preds).items())
        print(Counter(graph_labels).items())

    # wandb.log({"morphology confusion matrix": wandb.plot.confusion_matrix(y_true=np.array(node_labels),
    #                                                                       preds=np.array(node_preds),
    #                                                                       class_names=morphology_classes)})
    # wandb.log({"distribution confusion matrix": wandb.plot.confusion_matrix(y_true=np.array(graph_labels),
    #                                                                         preds=np.array(graph_preds),
    #                                                                         class_names=distribution_classes)})

    return node_acc, node_auc, graph_acc, graph_auc, val_node_losses.avg, val_graph_losses.avg, val_losses.avg

def main():

    global args

    norm_transform = Normalize(
        mean=[106.95998],
        std=[51.748093]
    )

    # graph_transform = T.Compose([
    #     # T.KNNGraph(k=4),
    #     T.RadiusGraph(r=112), # 196, 56
    #     T.Cartesian(),
    #     # T.Distance(),
    #     # T.GDC(self_loop_weight=1, normalization_in='sym',
    #     #       normalization_out='col',
    #     #       diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #     #       sparsification_kwargs=dict(method='topk', k=4,dim=0),
    #     #       exact=True)
    #     # T.Polar()
    # ])

    graph_transforms = [
        T.Compose([
                T.RadiusGraph(r=112), # 196, 56
                T.Cartesian(),
            ]),
        T.Compose([
            T.KNNGraph(k=4),  # 196, 56
            T.Cartesian(),
        ]),
        # T.Compose([
        #     T.RadiusGraph(r=56),
        #     T.Polar(),
        # ])
    ]

    train_transform = Compose([
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=10, p=.5),
        Flip(p=.5),
        # RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4)
    ])

    train_data = _read_txt(args.train_txt)
    test_data = _read_txt(args.test_txt)

    data = [train_data[0] + test_data[0], train_data[1] + test_data[1]]
    id_list = _img2ids(data)
    first_ids = np.array([i for i in id_list.keys() if id_list[i][-1] != 1])
    first_y = np.array([id_list[i][-1] for i in id_list.keys() if id_list[i][-1] != 1])
    second_ids = np.array([i for i in id_list.keys() if id_list[i][-1] == 1])
    second_y = np.array([id_list[i][-1] for i in id_list.keys() if id_list[i][-1] == 1])
    skf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    for fold, (train_index, test_index) in enumerate(skf.split(first_ids, first_y)):
        # wandb.init(project='multi-task-cv', name=f'fold_{fold+1}')
        print("######### Fold {} #########".format(str(fold+1)))

        best_acc_morph = 0
        best_acc_dist = 0
        best_auc_morph = 0
        best_auc_dist = 0

        first_X_train, first_X_test = first_ids[train_index], first_ids[test_index]
        first_y_train, first_y_test = first_y[train_index], first_y[test_index]
        second_X_train, second_X_test, second_y_train, second_y_test = train_test_split(second_ids,
                                                                                        second_y, test_size=0.2,
                                                                                        random_state=fold)
        X_train_id = np.concatenate([first_X_train, second_X_train])
        X_test_id = np.concatenate([first_X_test, second_X_test])
        y_train_id = np.concatenate([first_y_train, second_y_train])
        y_test_id = np.concatenate([first_y_test, second_y_test])

        X_train, y_train = _ids2imgpaths(X_train_id, y_train_id, id_list)
        X_test, y_test = _ids2imgpaths(X_test_id, y_test_id, id_list)

        train_dataset = MultiTaskCalsListDataset(
            X=X_train,
            y=y_train,
            patch_size=args.patch_size,
            norm_transform=norm_transform
        )
        test_dataset = MultiTaskCalsListDataset(
            X=X_test,
            y=y_test,
            patch_size=args.patch_size,
            norm_transform=norm_transform
        )

        weights = balance_five_weights(args.train_txt, postoneg=[1.0, 1.0, 1.0, 1.0, 1.0])
        train_sampler = WeightedRandomSampler(weights, len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   # sampler=train_sampler,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

        total_step = len(train_loader)
        model = genconv_mg(num_features=args.num_features,
                           hidden_size=args.hidden_size,
                           node_num_classes=args.node_num_classes,
                           graph_num_classes=args.graph_num_classes,
                           n_layers=args.num_layers,
                           graph_transforms=graph_transforms,
                           patch_size=args.patch_size,
                           device=device
                           )
        model.to(device)

        Weights = Weight(tasks).to(device)
        params = []
        params += model.parameters()
        params += [Weights.weights]

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.pretrain:
            if os.path.isfile(args.pretrain_path):
                print("=> loading checkpoint '{}'".format(args.pretrain_path))
                checkpoint = torch.load(args.pretrain_path)

                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items()
                                   if k in model_dict and v.size() == model_dict[k].size()}

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=True)

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.pretrain_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrain_path))

        if args.restart:
            args.start_epoch = 0

        # wandb.watch(model, log="all", log_freq=10)

        initial_task_loss = None
        for epoch in range(args.start_epoch, args.num_epochs):

            adjust_learning_rate(optimizer, epoch, args)

            # Train for one epoch
            loss, Weights, initial_task_loss = train(train_loader, model, criterion1, criterion2, optimizer,
                                                     epoch, total_step, Weights, initial_task_loss)

            node_acc, node_auc, graph_acc, graph_auc, \
            val_node_loss, val_graph_loss, val_loss = validate(test_loader, model, criterion1, criterion2, Weights)

            # wandb.log({"Training Loss": loss, "Validation Loss": val_loss, "Validation Node Loss": val_node_loss,
            #            "Validation Graph Loss": val_graph_loss, "Node Accuracy": node_acc, "Node AUC": node_auc,
            #            "Graph Accuracy": graph_acc, "Graph AUC": graph_auc})

            if (epoch + 1) % 600 == 0:
                print("Saving epoch {} model. ".format(str(epoch + 1)))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc_morph': best_acc_morph,
                    'best_acc_dist': best_acc_dist,
                    'best_auc_morph': best_auc_morph,
                    'best_auc_dist': best_auc_dist,
                    'optimizer': optimizer.state_dict()
                }, is_best=0,
                    filename=os.path.dirname(args.model_path) + '/checkpoint_{}.pth.tar'.format(str(epoch + 1)))

            # remeber best accuracy model and save checkpoint
            # is_best = acc > best_acc1
            # Morphology
            is_best_morph = node_auc > best_auc_morph
            # best_acc1 = max(acc, best_acc1)
            best_auc_morph = max(node_auc, best_auc_morph)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_morph': best_acc_morph,
                'best_acc_dist': best_acc_dist,
                'best_auc_morph': best_auc_morph,
                'best_auc_dist': best_auc_dist,
                'optimizer': optimizer.state_dict()
            }, is_best_morph, filename=os.path.join(os.path.dirname(args.model_path), str(fold), 'morphology/checkpoint.pth.tar'))

            # Morphology
            is_best_dist = graph_auc > best_auc_dist
            # best_acc1 = max(acc, best_acc1)
            best_auc_dist = max(graph_auc, best_auc_dist)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_morph': best_acc_morph,
                'best_acc_dist': best_acc_dist,
                'best_auc_morph': best_auc_morph,
                'best_auc_dist': best_auc_dist,
                'optimizer': optimizer.state_dict()
            }, is_best_dist, filename=os.path.join(os.path.dirname(args.model_path), str(fold), 'distribution/checkpoint.pth.tar'))
        # wandb.join()
if __name__ == '__main__':
    set_random_seed()
    print(args)
    main()
