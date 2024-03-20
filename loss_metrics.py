import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score
from medpy.metric.binary import hd95


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class UncertainBCE(nn.Module):
    def __init__(self, epsilon=0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target, uncertainty):
        tmp_uncertain = uncertainty + self.epsilon

        loss = nn.BCELoss(weight=tmp_uncertain)(preds, target)
        return loss


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA)
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc  # e.g. np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def dice_coef(self):
        dice_up = 2 * np.diag(self.confusionMatrix)
        dice_down = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0)
        dice = dice_up / dice_down
        return dice

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]

        # count = np.bincount(label.cuda().data.cpu().numpy(), minlength=self.numClass ** 2)
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def sensitivity(self):
        # Sensitivity, recall, hit rate, or true positive rate (TPR)
        # TPR = TP / (TP + FN)
        TP = np.diag(self.confusionMatrix)
        FN = np.sum(self.confusionMatrix, axis=0) - TP
        sensitivity = TP / (TP + FN + 1e-10)  # Adding 1e-10 to avoid division by zero
        return sensitivity

    def specificity(self):
        # Specificity or true negative rate (TNR)
        # TNR = TN / (TN + FP)
        FP = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        TN = np.sum(self.confusionMatrix) - (FP + np.sum(self.confusionMatrix, axis=0))
        specificity = TN / (TN + FP + 1e-10)  # Adding 1e-10 to avoid division by zero
        return specificity

    def addBatch(self, imgPredict, imgLabel):
        # assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
