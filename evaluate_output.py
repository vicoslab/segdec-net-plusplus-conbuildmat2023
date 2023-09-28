import os.path
from pathlib import Path
import argparse
import cv2
import numpy as np

import torch

def FP(y_true, y_pred):
    return ((1 - y_true) * y_pred).sum()

def FN(y_true, y_pred):
    return (y_true*(1 - y_pred)).sum()

def precision(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_true))
    TP = (y_true * y_pred).sum()
    FP = ((one-y_true)*y_pred).sum()
    return (TP + 1e-15) / (TP + FP + 1e-15)

def general_precision(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return precision(y_true, y_pred)

def recall(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_pred))
    one = one.numpy()
    TP = (y_true * y_pred).sum()
    FN = (y_true*(one - y_pred)).sum()
    return (TP + 1e-15) / (TP + FN + 1e-15)

def general_recall(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return recall(y_true, y_pred)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

if __name__ == '__main__':
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-ground_truth_dir', type=str, help='path where ground truth images are located')
    parser.add_argument('-gt_filenames', type=str, default=None, help='txt with filenames for validation ', action='append')
    parser.add_argument('-pred_dir', type=str, default='prediction_output',  help='path with predictions')
    parser.add_argument('-threshold', type=float, default=0.2,  help='crack threshold detection')
    args = parser.parse_args()

    result_precision = []
    result_recall = []
    result_f1 = []
    result_dice = []
    result_jaccard = []

    cls_pred = []
    cls_gt = []

    if args.gt_filenames:
        img_names = []
        for data_filename in args.gt_filenames:
            assert os.path.exists(data_filename), f'{data_filename} does not exist'
            with open(data_filename) as f:
                img_names += [l.strip() for l in f.readlines()]

        paths = [Path(args.ground_truth_dir, f) for f in sorted(img_names)]
    else:
        paths = [path for path in  Path(args.ground_truth_dir).glob('*')]

    for file_name in paths:
        y_true = (cv2.imread(str(file_name), 0) > 128).astype(np.uint8)
        cls_gt.append(y_true.any())

        pred_file_name = Path(args.pred_dir) / file_name.name
        if not pred_file_name.exists():

            pred_file_name = Path(args.pred_dir) / (os.path.splitext(file_name.name)[0] + ".png")
            if not pred_file_name.exists():
                print(f'missing prediction for file {file_name.name} (.jpg or .png not found)')
                continue

        pred_image = cv2.imread(str(pred_file_name), 0)
        y_pred = (pred_image > 255 * args.threshold).astype(np.uint8)
        cls_pred.append(np.max(pred_image)/255.0)

        result_precision += [precision(y_true, y_pred).item()]
        if y_true.any(): # ignore images without cracks for recall
            result_recall += [recall(y_true, y_pred).item()]
        result_dice += [dice(y_true, y_pred).item()]
        result_jaccard += [jaccard(y_true, y_pred).item()]

    print('Precision = ', np.mean(result_precision), np.std(result_precision))
    print('recall = ', np.mean(result_recall), np.std(result_recall))
    print('f1 = ', 2*np.mean(result_precision)*np.mean(result_recall)/(np.mean(result_precision)+np.mean(result_recall)), 2*np.std(result_precision)*np.std(result_recall)/(np.std(result_precision)+np.std(result_recall)))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))

    if len(np.unique(cls_gt)) > 1:
        cls_gt = np.array(cls_gt)
        cls_pred = np.array(cls_pred)

        # run AP and AUC on scores
        from sklearn.metrics import average_precision_score, roc_auc_score
        print('CLASS ONLY: AP = ', average_precision_score(cls_gt, cls_pred))
        print('CLASS ONLY: AUC = ', roc_auc_score(cls_gt, cls_pred))

        # and run other metrics on binary output
        cls_pred = (cls_pred > args.threshold).astype(np.uint8)

        cls_precision = precision(cls_gt, cls_pred).item()
        cls_recall = recall(cls_gt, cls_pred).item()

        print('CLASS ONLY: Precision = ', cls_precision, 'FP =', FP(cls_gt, cls_pred))
        print('CLASS ONLY: recall = ', cls_recall, 'FN =', FN(cls_gt, cls_pred))
        print('CLASS ONLY: F1 = ', 2*cls_precision*cls_recall/(cls_precision+cls_recall))

