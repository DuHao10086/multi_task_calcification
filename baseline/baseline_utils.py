import torch
import shutil
import os
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

def _img2ids(data_list):
    result = {}
    for idx, img_path in enumerate(data_list[0]):
        dist_label = data_list[1][idx]
        pid = img_path.split("/")[-1].split("_")[0]
        if pid not in list(result.keys()):
            result[pid] = [[img_path], dist_label]
        else:
            result[pid][0].append(img_path)
    return result

def _ids2imgpaths(X, y, id_list):
    X_i = []
    y_i = []
    for i in X:
        X_i += id_list[i][0]
        y_i += [id_list[i][-1] for _ in id_list[i][0]]
    return X_i, y_i

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args, steps=30):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    lr = args.learning_rate * (0.1 ** (epoch // steps))
    for param_group in optimizer.param_groups:
        param_group['learning_rate'] = lr

def save_checkpoint(state, is_best, filename='models/alexnet/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Saving the best model. ")
        temp = os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        shutil.copyfile(filename, temp)


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def balance_weights(filepath, postoneg=[1.0, 1.0]):
    with open(filepath, 'r') as f:
        content = [int(i.strip().split()[1]) for i in f.readlines()]

    positives = np.array(content)==1
    pos_weight = postoneg[0]
    neg_weight = postoneg[1]
    sample_weights = np.zeros(len(positives))
    sample_weights[positives] = pos_weight /positives.sum()
    sample_weights[~positives] = neg_weight / (~positives).sum()
    return sample_weights

def balance_five_weights(content, postoneg=[1.0, 1.0, 1.0, 1.0, 1.0]):

    first = np.array(content)==0
    second = np.array(content)==1
    third = np.array(content)==2
    fourth = np.array(content)==3
    fifth = np.array(content)==4

    first_weight = postoneg[0]
    second_weight = postoneg[1]
    third_weight = postoneg[2]
    fourth_weight = postoneg[3]
    fifth_weight = postoneg[4]

    sample_weights = np.zeros(len(first))
    sample_weights[first] = first_weight /first.sum()
    sample_weights[second] = second_weight / second.sum()
    sample_weights[third] = third_weight / third.sum()
    sample_weights[fourth] = fourth_weight / fourth.sum()
    sample_weights[fifth] = fifth_weight / fifth.sum()
    return sample_weights

def ResizePad(img, size=(224, 224), gray=True):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    if gray:
        interp = cv2.INTER_NEAREST

    # aspect ratio of image
    aspect = float(w)/h
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    padColor = 0

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def Resize(img, size=(224, 224), interpolation=cv2.INTER_NEAREST):
    resized = cv2.resize(img, size, interpolation)
    return resized

def RescaleShape(shape, R=3, C=3, size=224):
    batch, channel, h, w = shape
    step = max(h//R, w//R)
    imglist = [[0, h, 0, w]]
    deltax = (step * R - w)//3
    deltay = (step * C - h)//3
    tempx = 0
    for i in range(C):
        nextx = tempx + step - deltax
        if nextx > w:
            nextx = w
        tempy = 0
        for j in range(R):
            nexty = tempy + step - deltay
            if nexty > h:
                nexty = h
            imglist.append([tempy, nexty, tempx, nextx])
            tempy = nexty
        tempx = nextx
    return imglist

def RescaleImage(shape, coord, R=3, C=3, size=224):
    h, w = shape
    step = max(h//R, w//R)
    imglist = [coord]
    deltax = (step * R - w)//3
    deltay = (step * C - h)//3
    tempx = coord[2]
    for i in range(C):
        nextx = tempx + step - deltax
        if nextx > coord[3]:
            nextx = coord[3]
        tempy = coord[0]
        for j in range(R):
            nexty = tempy + step - deltay
            if nexty > coord[1]:
                nexty = coord[1]
            imglist.append([tempy, nexty, tempx, nextx])
            tempy = nexty
        tempx = nextx
    return imglist

def IndexImage(img, idx):
    return img[idx[0]:idx[1], idx[2]:idx[3]]

def CreateShapedir(shape, R=3, C=3, size=224):
    shapedir = {}
    shapedir['1'] = RescaleShape(shape)
    shapedir['2'] = [RescaleImage((idx[1]-idx[0], idx[3]-idx[2]), idx) for idx in shapedir['1'][1:]]
    return shapedir

def create_class_weight(freq_tensor):
    total = freq_tensor.sum()
    class_weight = total / freq_tensor
    return class_weight.float()

def create_class_freq(cls_tensor, freq_tensor, num_class=5, theta=0.1):
    class_freq = []
    for i in range(num_class):
        if i in cls_tensor:
            idx = cls_tensor.tolist().index(i)
            class_freq.append(freq_tensor[idx].item())
        else:
            class_freq.append(theta)
    return class_freq

class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

# def balance_weights(df_source, col_target, mlb):
#     """
#     Compute balanced weights from a Multilabel dataframe
#     Compute balanced weights from a Multilabel dataframe
#
#     Arguments:
#         Dataframe
#         The name of the column with the target labels
#         A MultiLabelBinarizer to one-hot-encode/decode the label column
#
#     Returns:
#         A Pandas Series with balanced weights
#     """
#
#     # Create a working copy of the dataframe
#     df = df_source.copy(deep=True)
#
#     df_labels = mlb.transform(df[col_target].str.split(" "))
#
#     ## Next 4 lines won't be needed when axis argument is added to np.unique in NumPy 1.13
#     ncols = df_labels.shape[1]
#     dtype = df_labels.dtype.descr * ncols
#     struct = df_labels.view(dtype)
#     uniq_labels, uniq_counts = np.unique(struct, return_counts=True)
#
#     uniq_labels = uniq_labels.view(df_labels.dtype).reshape(-1, ncols)
#
#     ## We convert the One-Hot-Encoded labels as string to store them in a dataframe and join on them
#     df_stats = pd.DataFrame({
#         'target': np.apply_along_axis(np.array_str, 1, uniq_labels),
#         'freq': uniq_counts
#     })
#
#     df['target'] = np.apply_along_axis(np.array_str, 1, df_labels)
#
#     ## Join the dataframe to add frequency
#     df = df.merge(df_stats, how='left', on='target')
#
#     ## Compute balanced weights
#     weights = 1 / df['freq'].astype(np.float)
#
#     return weights