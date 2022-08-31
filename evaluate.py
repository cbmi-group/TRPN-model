
import os,sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
currentdir = os.path.dirname(__file__)
sys.path.append(currentdir)
from datasets.metrics import evaluate, hausdorff_distance

if __name__ == "__main__":

    data_list = './data/data_list/er_tubule_confocal_64/test.txt'
    results_dir = './data/results/er-test.txt/experiments_20210906/train_ranet_model-mito-train_oneshot_support_for_er.txt-2022516_1213/checkpoints_epoch_30'

    with open(data_list, 'r') as fid:
        lines = fid.readlines()
    img_num = len(lines)

    print(img_num)

    predictions = []
    labels = []    
    for i, l in enumerate(lines):
        img_dir = l.strip().split()[0]
        mask_dir = img_dir.replace('images', 'masks')
        img_name = os.path.join(results_dir, os.path.splitext(os.path.split(img_dir)[-1])[0]+".npy")

        if not os.path.exists(img_dir):
            print("==> Work on %s." % (img_name))
            continue
        
        result = np.load(img_name)
        mask = cv2.imread(mask_dir, 0)
        mask[mask==255] = 1

        predictions.append(result.flatten())
        labels.append(mask.flatten())

    y_true, y_pred = np.concatenate(labels, axis=0), np.concatenate(predictions, axis=0)

    best_threshold, best_f1, best_iou, best_acc, best_spc, best_sen, auc = evaluate(y_pred, y_true, mode='f1', is_auc=True)

    print("Best Threshold: %.2f. Best F1 Score: %.4f, Best IoU: %.4f, AUC: %.4f, SPC: %.4f, SEN: %.4f, ACC: %.4f." \
        % (best_threshold, best_f1, best_iou, auc, best_spc, best_sen, best_acc))








    
