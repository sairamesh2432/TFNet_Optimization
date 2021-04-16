import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


class ROC_PR():
    def __init__(self, pred, label):
        '''
        Draw mean and individual ROC & PR
        :param pred: predictions, numpy array, n_label * n_sample
        :param label: labels, same format as pred`
        '''
        self.pred  = pred
        self.label = label
        self.n     = label.shape[0]

    def ROC(self, title=""):
        '''
        :param title: title of figure
        :return: ROC plt and mean auc
        '''

        if self.n > 1:
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)

            for i in range(self.n):
                fpr, tpr, thresholds = roc_curve(self.label[i], self.pred[i])
                mean_tpr += np.interp(mean_fpr, fpr, tpr)  # interp
                plt.plot(fpr, tpr, lw=0.1)

            # plot mean roc
            mean_tpr /= self.n
            mean_auc = auc(mean_fpr, mean_tpr)  # mean auc
            plt.plot(mean_fpr, mean_tpr, lw=3, color=(0.2, 0.2, 0.2))
            title = title + '\n Mean AUROC = ' + str(mean_auc)

        else:
            fpr, tpr, thresholds = roc_curve(self.label[0], self.pred[0])
            mean_auc = auc(fpr, tpr)  # auc
            plt.plot(fpr, tpr, lw=3, color=(0.2, 0.2, 0.2))
            title = title + '\n AUROC = ' + str(mean_auc)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)

        return plt, mean_auc

    def PR(self, title=""):
        '''
        :param title: title of figure
        :return: PR plt and mean auc
        '''

        if self.n > 1:
            mean_precision = 0.0
            mean_recall = np.linspace(0, 1, 100)

            for i in range(self.n):
                precision, recall, thresholds = precision_recall_curve(self.label[i], self.pred[i])
                mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])  # interp
                plt.plot(recall, precision, lw=0.1)

            # plot mean roc
            mean_precision /= self.n
            mean_auc = auc(mean_recall, mean_precision)  # mean auc
            plt.plot(mean_recall, mean_precision, lw=3, color=(0.2, 0.2, 0.2))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            title = title + '\n Mean AUPR = ' + str(mean_auc)
            plt.title(title)

        else:
            precision, recall, thresholds = precision_recall_curve(self.label[0], self.pred[0])
            mean_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=3, color=(0.2, 0.2, 0.2))
            title = title + '\n AUPR = ' + str(mean_auc)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)

        return plt, mean_auc


