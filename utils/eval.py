# TODO:Evaluate model accuracy
import csv
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from prettytable import PrettyTable


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        """update confusion matrix"""
        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)
        mask = (labels >= 0) & (labels < self.num_classes)
        valid_preds = preds[mask]
        valid_labels = labels[mask]
        n = self.num_classes
        idx = n * valid_preds + valid_labels
        count = np.bincount(idx, minlength=n * n)
        confusion_matrix = count.reshape(n, n)
        self.matrix += confusion_matrix

    def summary(self, classes):
        mIoU, mF1 = 0, 0
        table = PrettyTable()
        table.field_names = ["Per-class", "IoU (%)", "F1-score (%)"]
        for i in range(classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            IoU = TP / (TP + FP + FN) if TP + FN + FP != 0 else 1.
            P = TP / (TP + FP) if (TP + FP) != 0 else 1.
            R = TP / (TP + FN) if (TP + FN) != 0 else 1.
            F1 = 2 * P * R / (P + R) if P + R != 0 else 0.
            table.add_row([self.labels[i], round(IoU * 100, 3), round(F1 * 100, 3)])
            mIoU += IoU
            mF1 += F1
        logging.info(f'Classes acc:\n{table}')
        mIoU = round(mIoU * 100 / classes, 3)
        mF1 = round(mF1 * 100 / classes, 3)
        OA = round(np.diag(self.matrix).sum() / self.matrix[:classes, :classes].sum() * 100, 3)
        logging.info(f'\tmIoU: {mIoU} %')
        logging.info(f'\tMacro F1-score: {mF1} %')
        logging.info(f'\tOA: {OA} %')
        return mIoU, mF1, OA

    def plot(self, save_path, n_classes):
        """plot confusion matrix"""
        matrix = self.matrix / (self.matrix.sum(0).reshape(1, n_classes) + 1e-8)  # normalize by classes
        # matrix = self.matrix / self.matrix.sum()
        plt.figure(figsize=(12, 9), tight_layout=True)
        sns.set(font_scale=1.0 if n_classes < 50 else 0.8)
        sns.heatmap(data=matrix,
                    annot=n_classes < 30,
                    annot_kws={"size": 12},
                    cmap='Blues',
                    fmt='.3f',
                    square=True,
                    linewidths=.16,
                    linecolor='white',
                    xticklabels=self.labels,
                    yticklabels=self.labels)

        plt.xlabel('True', size=16, color='purple')
        plt.ylabel('Predicted', size=16, color='purple')
        plt.title('Normalized confusion matrix', size=20, color='blue')
        plt.savefig(save_path + ".png", dpi=150)
        plt.close()

        # save as .csv file
        with open(save_path + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + self.labels)
            for i, row in enumerate(self.matrix):
                writer.writerow([self.labels[i]] + list(row))


def eval_net(net, loader, device, json_label_path, n_img, bg):
    assert os.path.exists(json_label_path), 'Cannot find {} file'.format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for label, _ in class_indict.items()]
    n_classes = 2 if net.n_classes == 1 else net.n_classes
    confusion = ConfusionMatrix(num_classes=n_classes, labels=labels)
    net.eval()

    with torch.no_grad():
        for batch in loader:
            images = []
            for n in range(1, 1 + n_img):
                imgs = batch[f'image{n}']
                imgs = imgs.to(device=device, dtype=torch.float32)
                images.append(imgs)

            true_masks = batch['mask']
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            pred_masks = net(images)
            images.clear()

            for true_mask, pred in zip(true_masks, pred_masks):
                true_mask = true_mask.squeeze(dim=0).cpu().numpy().flatten()

                if net.n_classes > 1:
                    pred = F.softmax(pred, dim=0)
                    pred = torch.argmax(pred, dim=0)
                    pred = pred.cpu().numpy().flatten()
                else:
                    pred = (pred > 0.5).int().squeeze(dim=0).cpu().numpy().flatten()

                # ignore background pixels
                if not bg:
                    mask = true_mask != (net.n_classes - 1)
                    pred, true_mask = pred[mask], true_mask[mask]
                    n_classes = net.n_classes - 1

                confusion.update(pred, true_mask)
        mIoU, mF1, OA = confusion.summary(n_classes)
    return mIoU, mF1, OA, confusion, n_classes
