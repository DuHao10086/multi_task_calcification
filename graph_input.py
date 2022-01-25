import numpy as np

from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import ImageEnhance
import cv2
import os
import torch
import skimage
import matplotlib.image as mpimg
import joblib
from input_utils import get_morph_mask
# from input_utils import norm_detect, add_img_margins

morph_class = ['not suspicious', 'amorphous', 'heterogeneous', 'pleomorphic', 'linear']

def find_annotation(img_path):
    annot_path = img_path.replace('.png', '_mask.png')
    annot = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
    return annot

def _read_txt(txt_path):
    data = open(txt_path, 'r').read().splitlines()
    imagepaths = []
    labels = []
    for d in data:
        imagepaths.append(d.split()[0])
        labels.append(int(d.split()[1]))
    return [imagepaths, labels]

def pad_img(img, patch_size=14):
    result = np.zeros((patch_size, patch_size))
    ht, wd= img.shape
    xx = (patch_size - wd) // 2
    yy = (patch_size - ht) // 2
    result[yy:yy+ht, xx:xx+wd] = img
    return result

def pad_img_rgb(img, patch_size=14):
    result = np.zeros((patch_size, patch_size, 3))
    ht, wd= img.shape[:2]
    xx = (patch_size - wd) // 2
    yy = (patch_size - ht) // 2
    result[yy:yy+ht, xx:xx+wd] = img
    return result

class MultiTaskCalsDataset(Dataset):
    def __init__(self, txt_path, patch_size, norm_transform=None,
                 morph_dir="/home/fathomx/workspace/cals_morp_mask"):
        self.data = _read_txt(txt_path)
        self.images_arr = np.asarray(self.data[0])
        self.labels_arr = np.asarray(self.data[1])
        self.patch_size = patch_size
        self.norm_transform = norm_transform
        self.data_len = len(self.data[0])
        self.morph_dir = morph_dir

    def __getitem__(self, index):
        img_path = self.images_arr[index]
        dist_label = self.labels_arr[index]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = find_annotation(img_path)

        height, width = img.shape

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        patches = []
        coords = []
        morph_labels = []

        morph_masks = get_morph_mask(img_path, morph_class, morph_dir=self.morph_dir)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            center = np.array([cx/width, cy/height])
            # center = np.array([cx, cy])
            rect_patch = img[cy - self.patch_size // 2:cy + self.patch_size // 2,
                         cx - self.patch_size // 2:cx + self.patch_size // 2]
            if rect_patch.shape != (self.patch_size, self.patch_size):
                rect_patch = pad_img(rect_patch, patch_size=self.patch_size)

            patches.append(rect_patch)
            coords.append(center)

            temp_node_labels = []
            for (morph_label, morph_mask) in morph_masks:
                rect_morph_mask = morph_mask[cy - self.patch_size // 2:cy + self.patch_size // 2,
                                  cx - self.patch_size // 2:cx + self.patch_size // 2]
                #         rect_morph_mask = morph_mask[y:y+h, x:x+h]
                if (rect_morph_mask != 0).sum() != 0:
                    node_label = morph_label
                else:
                    node_label = 0
                temp_node_labels.append(node_label)

            morph_labels.append(max(temp_node_labels))
        patches = np.stack(patches, axis=0)
        normed_patches = patches[:, np.newaxis, :, :]

        if self.norm_transform is not None:
            normed_patches = self.norm_transform(image=normed_patches)['image']

        normed_patches = normed_patches.squeeze()
        coords = np.stack(coords, axis=0)

        patch_tensor = torch.from_numpy(normed_patches[np.newaxis, :]).float()
        coord_tensor = torch.from_numpy(coords[np.newaxis, :]).float()
        morph_tensor = torch.from_numpy(np.array(morph_labels)).long()

        return patch_tensor, coord_tensor, morph_tensor, dist_label

    def __len__(self):
        return self.data_len


class MultiTaskCalsListDataset(Dataset):
    def __init__(self, X, y, patch_size, norm_transform=None,
                 morph_dir="/home/fathomx/workspace/cals_morp_mask"):
        self.images_arr = np.asarray(X)
        self.labels_arr = np.asarray(y)
        self.patch_size = patch_size
        self.norm_transform = norm_transform
        self.data_len = len(self.images_arr)
        self.morph_dir = morph_dir

    def __getitem__(self, index):
        img_path = self.images_arr[index]
        dist_label = self.labels_arr[index]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = find_annotation(img_path)

        height, width = img.shape

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        patches = []
        coords = []
        morph_labels = []

        morph_masks = get_morph_mask(img_path, morph_class, morph_dir=self.morph_dir)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            center = np.array([cx/width, cy/height])
            # center = np.array([cx, cy])
            rect_patch = img[cy - self.patch_size // 2:cy + self.patch_size // 2,
                         cx - self.patch_size // 2:cx + self.patch_size // 2]
            if rect_patch.shape != (self.patch_size, self.patch_size):
                rect_patch = pad_img(rect_patch, patch_size=self.patch_size)

            patches.append(rect_patch)
            coords.append(center)

            temp_node_labels = []
            for (morph_label, morph_mask) in morph_masks:
                rect_morph_mask = morph_mask[cy - self.patch_size // 2:cy + self.patch_size // 2,
                                  cx - self.patch_size // 2:cx + self.patch_size // 2]
                #         rect_morph_mask = morph_mask[y:y+h, x:x+h]
                if (rect_morph_mask != 0).sum() != 0:
                    node_label = morph_label
                else:
                    node_label = 0
                temp_node_labels.append(node_label)

            morph_labels.append(max(temp_node_labels))
        patches = np.stack(patches, axis=0)
        normed_patches = patches[:, np.newaxis, :, :]

        if self.norm_transform is not None:
            normed_patches = self.norm_transform(image=normed_patches)['image']

        normed_patches = normed_patches.squeeze()
        coords = np.stack(coords, axis=0)

        patch_tensor = torch.from_numpy(normed_patches[np.newaxis, :]).float()
        coord_tensor = torch.from_numpy(coords[np.newaxis, :]).float()
        morph_tensor = torch.from_numpy(np.array(morph_labels)).long()

        return patch_tensor, coord_tensor, morph_tensor, dist_label

    def __len__(self):
        return self.data_len


class MultiTaskCalsFloatDataset(Dataset):
    def __init__(self, X, y, patch_size, norm_transform=None):
        self.images_arr = np.asarray(X)
        self.labels_arr = np.asarray(y)
        self.patch_size = patch_size
        self.norm_transform = norm_transform
        self.data_len = len(self.images_arr)

    def __getitem__(self, index):
        img_path = self.images_arr[index]
        dist_label = self.labels_arr[index]

        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = mpimg.imread(img_path)
        if img.shape[-1] == 4:
            img = skimage.color.rgba2rgb(img)
        img = skimage.img_as_float32(img)

        mask = find_annotation(img_path)

        height, width = img.shape[:2]

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        patches = []
        coords = []
        morph_labels = []

        morph_masks = get_morph_mask(img_path, morph_class)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            center = np.array([cx/width, cy/height])
            # center = np.array([cx, cy])
            rect_patch = img[cy - self.patch_size // 2:cy + self.patch_size // 2,
                         cx - self.patch_size // 2:cx + self.patch_size // 2, :]
            if rect_patch.shape != (self.patch_size, self.patch_size, 3):
                rect_patch = pad_img_rgb(rect_patch, patch_size=self.patch_size)

            patches.append(rect_patch)
            coords.append(center)

            temp_node_labels = []
            for (morph_label, morph_mask) in morph_masks:
                rect_morph_mask = morph_mask[cy - self.patch_size // 2:cy + self.patch_size // 2,
                                  cx - self.patch_size // 2:cx + self.patch_size // 2]
                #         rect_morph_mask = morph_mask[y:y+h, x:x+h]
                if (rect_morph_mask != 0).sum() != 0:
                    node_label = morph_label
                else:
                    node_label = 0
                temp_node_labels.append(node_label)

            morph_labels.append(max(temp_node_labels))
        patches = np.stack(patches, axis=0)
        normed_patches = patches

        if self.norm_transform is not None:
            normed_patches = self.norm_transform(image=normed_patches)['image']

        normed_patches = normed_patches.squeeze()
        coords = np.stack(coords, axis=0)

        patch_tensor = torch.from_numpy(normed_patches[np.newaxis, :]).float()
        coord_tensor = torch.from_numpy(coords[np.newaxis, :]).float()
        morph_tensor = torch.from_numpy(np.array(morph_labels)).long()

        return patch_tensor, coord_tensor, morph_tensor, dist_label

    def __len__(self):
        return self.data_len