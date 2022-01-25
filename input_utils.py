import cv2
import os
import json
import numpy as np
import scipy
import torch
import gc
from PIL import Image, ImageDraw
from sklearn.preprocessing import normalize, scale

def row_normalization(features):
    ## normalize the feature matrix by its row sum
    rowsum = features.sum(dim=1)
    inv_rowsum = torch.pow(rowsum, -1)
    inv_rowsum[torch.isinf(inv_rowsum)] = 0.
    features = features * inv_rowsum[..., None]

    return features

def add_img_margins(img, margin_size):
    """ Add zero margins to an image """
    new_img = np.zeros((img.shape[0]+margin_size*2,
                        img.shape[1]+margin_size*2))
    new_img[margin_size:margin_size+img.shape[0],
            margin_size:margin_size+img.shape[1]] = img

    return new_img

def norm_detect(uint8_img_path):
    src = uint8_img_path
    # src = orig_img.copy()
    # src = enhanced_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Contrast enhancement
    dilated = cv2.dilate(src, kernel)
    topHat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    accentuated = cv2.add(src, topHat)
    highContrast = cv2.subtract(accentuated, blackHat)

    # Background subtraction
    backgroundSubtraction = cv2.subtract(highContrast, dilated)

    th, thresholding = cv2.threshold(backgroundSubtraction, 40, 255, cv2.THRESH_BINARY)
    return thresholding

def graph_features(img, xml_masks, avg_values, coord, n_ch=3):
    result_masks = []
    todel = []
    for node_idx, node_mask in enumerate(xml_masks):
        mask = node_mask == 1.0
        for c in range(n_ch):
            avg_values[node_idx, c] = np.mean(img[:, :, c][mask])
        coord[node_idx] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col

        if np.isnan(avg_values[node_idx].sum()) or np.isnan(coord[node_idx].sum()):
            todel.append(node_idx)
        result_masks.append(mask)
    avg_values_removed = np.delete(avg_values, todel, axis=0)
    coord_removed = np.delete(coord, todel, axis=0)
    result_masks_removed = [i for idx, i in enumerate(result_masks) if idx not in todel]
    return avg_values_removed, coord_removed, result_masks_removed

# @profile
def find_contour_mask(mask):
    graymask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(graymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    testmasks = []
    for cnt in contours:
        temp_mask = np.zeros(graymask.shape)
        cv2.drawContours(temp_mask,[cnt], 0, (255,255,255), -1)
        temp_mask[temp_mask!=0]=1.0
        testmasks.append(temp_mask)
        del temp_mask
        gc.collect()
    return testmasks

def find_rect_contour_mask(mask, patch_size=14):
    graymask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(graymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    testmasks = []
    for cnt in contours:
        temp_mask = np.zeros(graymask.shape)
        x,y,w,h = cv2.boundingRect(cnt)
        topx = (x+patch_size) if (x+patch_size)<graymask.shape[1] else graymask.shape[1]
        topy = (y+patch_size) if (y+patch_size)<graymask.shape[0] else graymask.shape[0]
        temp_mask[y:topy, x:topx] = 1.0
        testmasks.append(temp_mask)
        del temp_mask
        gc.collect()
    return testmasks

def patch_graph_features(img, xml_masks, patch_values, coord, n_ch, patch_size):
    result_masks = []
    todel = []
    for node_idx, node_mask in enumerate(xml_masks):
        mask = node_mask == 1.0
        temp = img[mask]
        if img[mask].shape[0]!=(patch_size*patch_size):
            temp = cv2.resize(temp, (patch_size, patch_size))
        else:
            temp = temp.reshape(patch_size, patch_size)
        temp = normalize(temp)
        patch_values[node_idx, :, :] = temp
        coord[node_idx] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col

        if np.isnan(patch_values[node_idx].sum()) or np.isnan(coord[node_idx].sum()):
            todel.append(node_idx)
        result_masks.append(mask)
    patch_values_removed = np.delete(patch_values, todel, axis=0)
    coord_removed = np.delete(coord, todel, axis=0)
    result_masks_removed = [i for idx, i in enumerate(result_masks) if idx not in todel]
    return patch_values_removed, coord_removed, result_masks_removed

def get_morph_mask(img_path, morph_class, morph_dir="/home/fathomx/workspace/cals_morp_mask"):

    morph_path = os.path.join(morph_dir,
                              os.path.basename(img_path).replace('png', 'json'))
    with open(morph_path, 'r') as f:
        morph_annot = json.load(f)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    masks = []
    for obj in morph_annot['outputs']['object']:
        label = morph_class.index(obj['name'])

        if 'bndbox' in obj.keys():
            morph_mask = np.zeros(img.shape).astype(np.uint8)
            annotations = obj['bndbox']
            morph_mask[annotations['ymin']:annotations['ymax'],
            annotations['xmin']:annotations['xmax']] = 255

        elif 'polygon' in obj.keys():
            annotations = list(obj['polygon'].values())
            mask = np.zeros(img.shape)
            morph_mask = Image.fromarray(mask)
            ImageDraw.Draw(morph_mask).polygon(annotations, outline=255, fill=255)
        else:
            print("WTF!")
        masks.append([label, np.array(morph_mask).astype(np.uint8)])
    return masks