import os
import glob
import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(segmentation, mask):
    n_imgs = segmentation.shape[0]
    total_dist = 0

    for i in range(n_imgs):
        non_zero_seg = np.transpose(np.nonzero(segmentation[i,...]))
        non_zero_mask = np.transpose(np.nonzero(mask[i,...]))
        h_dist = max(directed_hausdorff(non_zero_seg, non_zero_mask)[0], directed_hausdorff(non_zero_mask, non_zero_seg)[0])
        total_dist += h_dist

    mean_dist = total_dist / n_imgs

    return mean_dist


def evaluate(y_scores, y_true, interval=0.01, mode='f1', is_auc=False, verbose=True):
    # metric=['f1_score', 'iou', 'acc', 'sen', 'spec', 'auc']

    thresholds = np.arange(0,0.9,interval)
    acc = np.zeros(len(thresholds))
    specificity = np.zeros(len(thresholds))
    sensitivity = np.zeros(len(thresholds))
    precision = np.zeros(len(thresholds))
    f1 = np.zeros(len(thresholds))
    iou = np.zeros(len(thresholds))
    y_true.astype(np.int8)

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.int8)

        sum_area = (y_pred + y_true)
        sub_area = (y_pred - y_true)
        tp = float(np.sum(sum_area==2))
        tn = float(np.sum(sum_area==0))
        fp = float(np.sum(sub_area==1))
        fn = float(np.sum(sub_area==-1))
        acc[indy] = (tp + tn) / (tp + fn + fp + tn + 1e-10)
        sensitivity[indy] = tp / (tp + fn + 1e-10)
        specificity[indy] = tn / (tn + fp + 1e-10)
        precision[indy] = tp / (tp + fp + 1e-10)
        f1[indy] = 2 * sensitivity[indy] * precision[indy] / (sensitivity[indy] + precision[indy] + 1e-10)
        union = np.sum(sum_area == 1)
        iou[indy] = tp / float(union + tp)
        if verbose:
            print('threshold {:.10f} ==> f1 score: {:.4f}, iou: {:.4f}, acc: {:.4f}, sen: {:.4f}, spec: {:.4f}'.format(threshold,f1[indy], iou[indy], acc[indy], sensitivity[indy], specificity[indy]))

    if mode == 'f1':
        thred_indx = np.argmax(f1)
    else:
        thred_indx = np.argmax(iou)
    m_iou = iou[thred_indx]
    m_f1 = f1[thred_indx]
    m_acc = acc[thred_indx]
    m_spc = specificity[thred_indx]
    m_sen = sensitivity[thred_indx]

    if is_auc:
        m_auc = roc_auc_score(y_true, y_scores)
        return thresholds[thred_indx], m_f1, m_iou, m_acc, m_spc, m_sen, m_auc
    else:
        return thresholds[thred_indx], m_f1, m_iou, m_acc, m_spc, m_sen


def evaluate_multi_class(predictions, groundtruths):
    labels = np.unique(groundtruths)
    acc = np.zeros(2)
    specificity = np.zeros(2)
    sensitivity = np.zeros(2)
    precision = np.zeros(2)
    f1 = np.zeros(2)
    iou = np.zeros(2)
    indy = 0
    for l in [1,2]: # only compute foreground
        y_pred = (predictions==l).astype(np.int8)
        y_true = (groundtruths==l).astype(np.int8)

        sum_area = (y_pred + y_true)
        sub_area = (y_pred - y_true)
        tp = float(np.sum(sum_area==2))
        tn = float(np.sum(sum_area==0))
        fp = float(np.sum(sub_area==1))
        fn = float(np.sum(sub_area==-1))
        acc[indy] = (tp + tn) / (tp + fn + fp + tn + 1e-10)
        sensitivity[indy] = tp / (tp + fn + 1e-10)
        specificity[indy] = tn / (tn + fp + 1e-10)
        precision[indy] = tp / (tp + fp + 1e-10)
        f1[indy] = 2 * sensitivity[indy] * precision[indy] / (sensitivity[indy] + precision[indy] + 1e-10)
        union = np.sum(sum_area == 1)
        iou[indy] = tp / float(union + tp)
        indy += 1
    mean_f1 = np.mean(f1)
    mean_iou = np.mean(iou)
    mean_acc = np.mean(acc)
    mean_spc = np.mean(specificity)
    mean_sen = np.mean(sensitivity)

    results = {'single': [f1, iou, specificity, sensitivity, acc],
               'mean':   [mean_f1, mean_iou, mean_spc, mean_sen, mean_acc]}
    
    return results


def evaluate_single_class(predictions, groundtruths):
    y_pred = predictions.astype(np.int8)
    y_true = groundtruths.astype(np.int8)

    sum_area = (y_pred + y_true)
    sub_area = (y_pred - y_true)
    tp = float(np.sum(sum_area==2))
    tn = float(np.sum(sum_area==0))
    fp = float(np.sum(sub_area==1))
    fn = float(np.sum(sub_area==-1))
    acc = (tp + tn) / (tp + fn + fp + tn + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-10)
    union = np.sum(sum_area == 1)
    iou = tp / float(union + tp)
    
    return f1, iou, acc, specificity, sensitivity